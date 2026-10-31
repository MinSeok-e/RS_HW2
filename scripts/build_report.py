"""Render a Korean PDF from recorded experiments and actual Kaggle evidence."""
import argparse
import csv
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/minseok/data/semester/1/RS/RS_HW2'))
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Run PDF rendering on a compute node')
    root = args.root.resolve()
    run = Path(json.loads((root / 'results/latest_experiment.json').read_text())['run_dir'])
    result = json.loads((run / 'summary.json').read_text())
    if result['smoke'] or result['status'] != 'completed':
        raise ValueError('A completed full experiment is required')
    manifest = json.loads((run / 'manifest.json').read_text())
    data, ease, neumf = result['data'], result['ease'], result['neumf']
    ledger_file = root / 'submissions/kaggle_records.json'
    ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {}
    records = {m: ledger.get(result['run_id'] + ':' + m, {}) for m in ('ease', 'neumf')}
    report = root / 'report'
    report.mkdir(exist_ok=True)
    shots = {m: report / 'screenshots' / f'{m}.png' for m in records}
    complete = all(records[m].get('public_score') not in (None, '') and shots[m].exists() for m in records)
    basename = 'RS_HW2_2026713004_Minseok_Jeong' + ('' if complete else '_DRAFT')
    font_dir = Path('/home/minseok/.cache/rs-hw2-fonts')
    pdfmetrics.registerFont(TTFont('Nanum', str(font_dir / 'NanumGothic-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('NanumBold', str(font_dir / 'NanumGothic-Bold.ttf')))
    pdfmetrics.registerFontFamily('Nanum', normal='Nanum', bold='NanumBold', italic='Nanum', boldItalic='NanumBold')
    normal = ParagraphStyle('body', fontName='Nanum', fontSize=9.5, leading=16, spaceAfter=7, wordWrap='CJK')
    small = ParagraphStyle('small', parent=normal, fontSize=8, leading=12, spaceAfter=3)
    heading = ParagraphStyle('heading', parent=normal, fontName='NanumBold', fontSize=14, leading=20, spaceBefore=12, spaceAfter=9)
    title = ParagraphStyle('title', parent=heading, fontSize=19, leading=26, alignment=TA_CENTER, spaceBefore=0)
    story = []
    def p(text, style=normal):
        story.append(Paragraph(text, style))
    def h(text):
        p(text, heading)
    def table(rows, widths):
        wrapped = [[Paragraph(escape(str(v)), small) for v in row] for row in rows]
        t = Table(wrapped, colWidths=[w * mm for w in widths], repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8eef5')),
                               ('LINEBELOW', (0,0), (-1,0), 0.8, colors.HexColor('#61718a')),
                               ('LINEBELOW', (0,-1), (-1,-1), 0.6, colors.HexColor('#61718a')),
                               ('VALIGN', (0,0), (-1,-1), 'TOP'),
                               ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                               ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t, Spacer(1, 8)])
    def picture(path, width=150, max_height=95):
        image = Image(str(path))
        scale = min(width * mm / image.imageWidth, max_height * mm / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        story.extend([image, Spacer(1, 7)])
    def public(model):
        value = records[model].get('public_score')
        return str(value) if value not in (None, '') else '점수 확인 대기'
    def kaggle_evidence(model):
        p(f'<b>{model.upper()} public leaderboard NDCG@20: {escape(public(model))}</b>')
        if records[model].get('submission_id'):
            p(f'공식 제출 ID: {records[model]["submission_id"]}; 상태: {escape(records[model].get("status", ""))}', small)
        if shots[model].exists():
            picture(shots[model], max_height=105)
        else:
            p('<b>최종 제출 전 추가:</b> 해당 모델의 실제 Kaggle 점수가 보이는 웹사이트 스크린샷.', small)

    p('Introduction to Recommender Systems<br/>Homework #2', title)
    p('2026713004 / Minseok Jeong', ParagraphStyle('author', parent=normal, alignment=TA_CENTER))
    p('<link href="https://github.com/MinSeok-e/RS_HW2" color="blue">GitHub: https://github.com/MinSeok-e/RS_HW2</link>')
    if not complete:
        p('<b>보고서 초안:</b> 실험 결과와 공식 API 점수를 기록했다. 표시된 스크린샷/점수 항목을 채운 뒤 최종 제출한다.')
    h('공통 데이터 처리와 검증')
    p(f'대회가 제공한 train.csv의 {data["interactions"]:,}개 상호작용(사용자 {data["users"]:,}명, 영화 {data["items"]:,}개)만 사용했다. 원본 MovieLens와 외부 데이터는 사용하지 않았다. test.csv의 {data["test_users"]}명에게 각각 영화 20개를 추천한다.')
    p('모든 평점을 관측 여부 1로 변환했다. 낮은 평점도 제거하지 않았으며 평점의 크기는 입력 가중치나 관련성 가중치로 사용하지 않았다. 미관측 상호작용은 확정적인 비선호가 아닌 미관측으로 해석한다.')
    p('사용자마다 time_idx 오름차순으로 마지막 ceil(0.2 × n)개를 검증에 두고 나머지를 학습에 사용했다. 모든 설정에 동일한 분할을 적용했다. time_idx는 분할에만 사용하며, 두 모델 자체는 순차 추천 모델이 아니다.')
    p(f'학습 {data["fit_interactions"]:,}건, 검증 {data["validation_interactions"]:,}건, 검증 사용자 {data["validation_users"]}명이다. 학습에서 등장한 영화 {data["fit_active_items"]:,}개 중 사용자의 학습 영화를 제외한 모든 후보를 평가했다. 검증에만 등장한 영화 {data["validation_only_items"]:,}개의 정답 {data["cold_validation_interactions"]:,}건은 추천할 수 없지만 평가 분모에는 유지했다.')
    p('<b>NDCG@20:</b> 순위 r의 영화가 검증 정답이면 1/log2(r+1)을 더해 DCG를 계산하고, 정답을 앞에 배치한 이상적인 DCG(최대 20개)로 나눈다. 최종 점수는 사용자별 NDCG의 산술평균이다. 동점은 원래 movieId 오름차순으로 정한다.')
    p('Kaggle의 숨김 정답 분할과 관련성 세부 기준은 안내에 없으므로, 시간순 분할과 binary relevance는 명시적인 로컬 검증 가정이다. 로컬 검증 점수와 public leaderboard 점수를 구분한다.')
    table([['모델', 'Validation NDCG@20', 'Public NDCG@20'],
           ['인기순 참고 기준', f'{result["popularity_reference"]["ndcg@20"]:.6f}', '제출하지 않음'],
           ['EASE', f'{ease["ndcg@20"]:.6f}', public('ease')],
           ['NeuMF', f'{neumf["ndcg@20"]:.6f}', public('neumf')]], [48, 56, 56])

    story.append(PageBreak())
    h('Q1. Warm-Up: EASE')
    p('HW1의 EASE를 선택했다. 이진 사용자–영화 행렬 X에 대해 재구성 제곱오차와 L2 정규화의 합을 최소화하고, 아이템 가중치 행렬 B의 대각선을 0으로 제한한다. P = (X^T X + lambdaI)^(-1)일 때 B = I - P/diag(P)의 정확한 해를 사용했다.')
    p('사용자 수가 영화 수보다 작아 Woodbury 항등식으로 같은 해를 사용자 공간에서 계산했다. 작은 행렬에서 원래 아이템 공간 계산과 수치적으로 일치하는지 테스트했다.')
    with (run / 'ease_sweep.csv').open() as f:
        rows = list(csv.DictReader(f))
    table([['정규화 lambda', 'Validation NDCG@20', 'Validation Recall@20']] +
          [[f'{float(r["regularization"]):g}', f'{float(r["ndcg@20"]):.6f}', f'{float(r["recall@20"]):.6f}'] for r in rows], [35, 62, 63])
    p(f'검증 NDCG로 선택한 lambda는 {ease["regularization"]:g}이고 점수는 {ease["ndcg@20"]:.6f}였다. 탐색한 범위에서 정규화를 높이면서 성능이 개선되다가 lambda=1000에서는 하락했다. 선택한 lambda로 전체 train.csv를 다시 학습하여 제출 파일을 생성했다.')
    picture(run / 'ease_validation.png', width=140, max_height=85)
    kaggle_evidence('ease')

    story.append(PageBreak())
    h('Q2. Deep Learning-based CF: NeuMF')
    p('NeuMF를 PyTorch로 구현했다. GMF 가지는 사용자·영화 임베딩의 원소별 곱을 사용하고, MLP 가지는 별도 임베딩을 연결해 ReLU 층을 통과시킨다. 두 출력을 연결한 뒤 선형 logit에 BCEWithLogitsLoss를 적용했다. 사전학습 없이 시드 2020으로 초기화했다.')
    p('학습 positive마다 그 사용자의 학습 영화를 제외한 fit-active 영화에서 negative를 균일 복원추출했다. Negative sampling은 검증 정답을 조회하지 않는다. Adam, batch size 2048, 최대 100 epoch를 사용하고 검증 NDCG가 10 epoch 연속 개선되지 않으면 중단했다.')
    trials = json.loads((run / 'neumf_sweep.json').read_text())
    table([['실험', 'GMF', 'MLP', 'LR', 'Neg', 'WD', 'Best epoch', 'NDCG@20']] +
          [[r['trial'].split('_')[-1], r['config']['mf_dim'], r['config']['mlp_dim'], f'{r["config"]["learning_rate"]:g}',
            r['config']['negatives'], f'{r["config"]["weight_decay"]:g}', r['best_epoch'], f'{r["ndcg@20"]:.6f}'] for r in trials],
          [15, 15, 15, 20, 13, 20, 27, 35])
    p('GMF/MLP는 각 가지의 임베딩 크기, Neg는 positive당 negative 수, WD는 전체 파라미터에 적용한 Adam L2 weight decay이다. MLP 은닉층은 실험 02에서 [64,32,16], 나머지에서 [32,16,8]이다.', small)
    p(f'선택된 설정은 {neumf["trial"]}, 검증 NDCG@20은 {neumf["ndcg@20"]:.6f}, 최적 epoch는 {neumf["best_epoch"]}였다. 전체 train.csv로 새로 초기화하여 동일 설정과 시드로 {neumf["best_epoch"]} epoch 학습했다. 재학습 중 검증/테스트 성능으로 종료 시점을 바꾸지 않았다.')
    picture(run / 'neumf_validation.png', max_height=85)
    kaggle_evidence('neumf')

    story.append(PageBreak())
    h('비교와 해석')
    p('이번 시간순 검증과 하이퍼파라미터 탐색에서는 EASE가 NeuMF보다 높은 NDCG@20을 보였다. EASE는 전체 상호작용 구조의 선형 관계를 학습하며, NeuMF는 negative sampling으로 비선형 사용자–영화 함수를 학습한다. 데이터 규모와 최적화·정규화 조건이 결과에 영향을 줄 수 있다.')
    p('NeuMF에서 임베딩 크기를 늘리는 것이 항상 성능 개선으로 이어지지는 않았다. 선택된 작은 L2 페널티는 이번 실험에서 가장 높은 검증 성능을 보였다. 단일 시드와 분할 결과이므로 통계적 유의성이나 일반적인 모델 우열을 주장하지 않는다.')
    p('추가 개선 방향은 다른 시드·검증 구간에서의 안정성 확인, 평점 정보의 활용 비교, NeuMF의 GMF/MLP 사전학습, 더 넓은 정규화 탐색이다. Kaggle public 점수로 반복 튜닝하지 않고 검증으로 선택한 모델별 파일을 한 번씩 제출했다.')
    h('실행 증거와 재현')
    p(f'학습 코드 커밋: {result["source_commit"]}.<br/>실험 Slurm Job ID: {manifest["slurm"]["SLURM_JOB_ID"]}.<br/>GPU: {escape(manifest.get("gpu", {}).get("name", "unknown"))}.')
    p('Python 3.10.20, PyTorch 2.3.1/CUDA 12.1, NumPy 1.26.4를 사용했다. 원본 실행은 GPU 1개, CPU 4개, RAM 6GB를 요청했다. 코드 커밋을 별도 작업 폴더에 추출해 실행 중 편집과 실제 학습 소스를 분리했다.')
    p('수작업 NDCG, 학습 영화·비활성 영화 제외, 시간순 분할, EASE 수식 일치, negative sampling, NeuMF 학습, 제출 ID 매핑 등 9개 테스트가 통과했다. GPU smoke 실행 뒤 본실험을 진행했고, 저장 모델을 다시 불러와 두 최종 CSV가 바이트 단위로 동일하게 재현됨을 확인했다.')
    p('최종 CSV는 test.csv의 588개 id 순서와 userId 대응, 사용자별 고유 영화 20개, 원래 movieId 복원, 이미 관측한 영화 제외 검사를 통과했다. 모델별 공식 제출 ID와 점수는 submissions/kaggle_records.json에 보관한다.')
    p('results/에는 설정·환경·데이터 해시·커밋 번호·epoch별 로그·결과·그래프가 있고, logs/에는 Slurm 로그가 있다. README에 재실행 명령을 제공한다. 원본 데이터, 모델 체크포인트, 인증 정보는 Git 추적에서 제외했다.')
    h('참고 문헌')
    p('Harald Steck. Embarrassingly Shallow Autoencoders for Sparse Data. WWW 2019.<br/><link href="https://arxiv.org/abs/1905.03375" color="blue">https://arxiv.org/abs/1905.03375</link>', small)
    p('Xiangnan He et al. Neural Collaborative Filtering. WWW 2017.<br/><link href="https://arxiv.org/abs/1708.05031" color="blue">https://arxiv.org/abs/1708.05031</link>', small)
    p('NeuMF 원저자 구현(구조 참고):<br/><link href="https://github.com/hexiangnan/neural_collaborative_filtering" color="blue">https://github.com/hexiangnan/neural_collaborative_filtering</link>', small)
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Nanum', 8)
        canvas.setFillColor(colors.HexColor('#687386'))
        canvas.drawString(22*mm, 13*mm, 'RS HW2 · 2026713004 · Minseok Jeong')
        canvas.drawRightString(188*mm, 13*mm, str(doc.page))
        canvas.restoreState()
    pdf = report / (basename + '.pdf')
    SimpleDocTemplate(str(pdf), pagesize=(210*mm,297*mm), leftMargin=22*mm, rightMargin=22*mm,
                      topMargin=18*mm, bottomMargin=22*mm, title='RS HW2 - Minseok Jeong', author='Minseok Jeong').build(story, onFirstPage=footer, onLaterPages=footer)
    status = {'pdf': str(pdf), 'complete': complete, 'experiment': result['run_id'],
              'missing_public_scores': [m for m in records if records[m].get('public_score') in (None, '')],
              'missing_screenshots': [m for m in shots if not shots[m].exists()]}
    (report / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
