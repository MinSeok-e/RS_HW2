import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from rs_hw2.data import load_data
from rs_hw2.evaluation import ranking_metrics, topk, validate_submission, write_submission
from rs_hw2.models import NeuMF, ease_scores, sample_epoch


class CoreTests(unittest.TestCase):
    def test_ndcg_hand_calculation_and_macro_average(self):
        ranks = np.array([[8, 2, 9], [4, 3, 7]])
        metrics, _ = ranking_metrics(ranks, [np.array([2, 9]), np.array([4])], k=3)
        user0 = (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
        self.assertAlmostEqual(metrics['ndcg@3'], (user0 + 1) / 2)
        self.assertEqual(metrics['recall@3'], 1)

    def test_perfect_and_zero_ndcg(self):
        perfect, _ = ranking_metrics(np.array([[2, 3, 1]]), [np.array([2, 3])], k=3)
        zero, _ = ranking_metrics(np.array([[0, 1, 4]]), [np.array([2, 3])], k=3)
        self.assertEqual(perfect['ndcg@3'], 1)
        self.assertEqual(zero['ndcg@3'], 0)

    def test_seen_and_inactive_exclusion_with_stable_ties(self):
        scores = np.array([[100., 4., 4., 90., 4.]])
        seen = np.array([[True, False, False, False, False]])
        ranked = topk(scores, seen, np.array([1, 1, 1, 0, 1]), k=3)
        np.testing.assert_array_equal(ranked, [[1, 2, 4]])
        self.assertEqual(scores[0, 0], 100)

    def test_bad_scores_and_too_few_candidates_rejected(self):
        with self.assertRaises(ValueError):
            topk(np.array([[np.nan, 2]]), np.zeros((1, 2)), k=1)
        with self.assertRaises(ValueError):
            topk(np.array([[1., 2.]]), np.array([[1, 0]]), k=2)

    def test_ease_user_space_matches_original_item_space(self):
        x = np.array([[1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 1, 0]], dtype=float)
        for regularization in (0.1, 1., 100.):
            p = np.linalg.inv(x.T @ x + regularization * np.eye(x.shape[1]))
            b = -p / np.diag(p)[None, :]
            np.fill_diagonal(b, 0)
            scores, _ = ease_scores(x, regularization)
            np.testing.assert_allclose(scores, x @ b, rtol=1e-10, atol=1e-10)

    def test_negative_sampling_excludes_fit_positives_and_inactive_items(self):
        seen = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0]], dtype=bool)
        u, i = np.nonzero(seen)
        a = sample_epoch(u, i, seen, 8, np.random.default_rng(8))
        b = sample_epoch(u, i, seen, 8, np.random.default_rng(8))
        for aa, bb in zip(a, b):
            np.testing.assert_array_equal(aa, bb)
        uu, ii, yy = a
        self.assertEqual(int((yy == 0).sum()), len(u) * 8)
        self.assertFalse(seen[uu[yy == 0], ii[yy == 0]].any())
        self.assertTrue(seen[uu[yy == 1], ii[yy == 1]].all())
        self.assertFalse((ii == 4).any())

    def test_submission_preserves_test_id_to_user_mapping(self):
        data = SimpleNamespace(user_ids=np.array([10, 20]), item_ids=np.array([100, 500, 900, 1000]),
                               test_ids=np.array([0, 1]), test_users=np.array([1, 0]))
        seen = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'submission.csv'
            write_submission(path, data, np.array([[2, 3], [0, 1]]), seen, k=2)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0], {'id': '0', 'recommendations': '100 500'})
            self.assertEqual(rows[1], {'id': '1', 'recommendations': '900 1000'})
            path.write_text('id,recommendations\n0,900 1000\n1,100 500\n')
            with self.assertRaises(ValueError):
                validate_submission(path, data, seen, k=2)

    def test_split_is_chronological_disjoint_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'train.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['userId', 'movieId', 'rating', 'time_idx'])
                for user in (20, 10):
                    for t in (4, 1, 3, 0, 2):
                        writer.writerow([user, 100 + t * 10, 3.5, t])
            (root / 'test.csv').write_text('id,userId\n0,20\n1,10\n')
            (root / 'sample_submission.csv').write_text('id,recommendations\n0,100 110\n1,110 120\n')
            data = load_data(root, 0.2)
            self.assertEqual(len(data.train_rows), 8)
            self.assertEqual(len(data.valid_rows), 2)
            self.assertFalse(set(data.train_rows) & set(data.valid_rows))
            self.assertEqual(set(data.train_rows) | set(data.valid_rows), set(range(10)))
            self.assertTrue((data.times[data.valid_rows] == 4).all())
            np.testing.assert_array_equal(data.user_ids[data.test_users], [20, 10])
            self.assertEqual(data.summary()['validation_only_items'], 1)

    def test_neumf_has_separate_embeddings_and_trains(self):
        torch.manual_seed(7)
        model = NeuMF(3, 5, mf_dim=4, mlp_dim=4, hidden=(8, 4))
        self.assertIsNot(model.gmf_user.weight, model.mlp_user.weight)
        users = torch.tensor([0, 1, 2, 0, 1, 2])
        items = torch.tensor([0, 1, 2, 3, 4, 0])
        labels = torch.tensor([1., 1., 1., 0., 0., 0.])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        initial = loss_fn(model(users, items), labels).item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = loss_fn(model(users, items), labels)
            loss.backward()
            optimizer.step()
        self.assertLess(loss_fn(model(users, items), labels).item(), initial / 2)
        self.assertGreater(model.gmf_user.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.mlp_user.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
