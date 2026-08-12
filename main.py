"""
Mini NPU Simulator
==================
AI(NPU)가 이미지를 판별할 때 쓰는 MAC(Multiply-Accumulate) 연산을
표준 라이브러리만으로 직접 구현한 콘솔 애플리케이션.

- 모드 1: 사용자 입력(3x3) 필터 A/B, 패턴을 받아 MAC 점수/시간/판정 출력
- 모드 2: data.json 을 로드해 Cross/X 필터로 일괄 판정(PASS/FAIL) + 성능 분석 + 요약

외부 라이브러리(NumPy, pandas 등) 사용 금지. json, time 등 표준 라이브러리만 사용.
"""

import json 
import time
import os

# =============================================================
# 0. 상수 / 설정
# =============================================================

EPSILON = 1e-9          # 1e-9 는 '지수 표기법'으로 0.000000001 을 뜻합니다(10의 -9제곱).
                        # 두 점수가 이 값보다 덜 차이나면 '사실상 같다'고 볼 때 쓰는 기준.
CROSS = "Cross"         # 표준 라벨 1 
X = "X"                 # 표준 라벨 2
UNDECIDED = "UNDECIDED" # 동점이라 어느 쪽인지 판정할 수 없을 때 쓰는 이름

DATA_FILE = "data.json"

# 성능 측정 반복 횟수 (요구사항 최소 기준 10회 이상).
# 3x3 같은 소형 연산은 타이머 해상도 대비 너무 빨라 값이 0에 수렴하므로,
# 안정적인 평균을 얻기 위해 반복 횟수를 넉넉히 둔다. (10회 이상 조건 충족)
PERF_REPEAT = 2000
MODE1_REPEAT = 10       # 모드1 결과 예시의 "평균/10회" 라벨과 맞춤


# =============================================================
# 1. 라벨 정규화(표준화)
# =============================================================

def normalize_label(raw):
    """
    다양한 표기의 라벨을 표준 라벨(Cross / X)로 정규화한다.
      expected 값 : '+' -> Cross,  'x' -> X
      filter  키  : 'cross' -> Cross, 'x' -> X
    대소문자/공백은 무시한다. 알 수 없는 라벨이면 None 을 반환한다.
    """
    # raw 가 None(값 없음)이면 비교할 게 없으니 바로 None 을 돌려줍니다.
    if raw is None:
        return None
    #  Cross ", "CROSS" 등이 전부 "cross" 로 통일되어 비교가 쉬워집니다.
    s = str(raw).strip().lower()
    if s in ("+", "cross"):
        return CROSS
    # ("x",) 처럼 원소가 1개인 묶음(튜플)은 쉼표를 꼭 붙여야 합니다. ("x") 는 그냥 문자열이 됨.
    if s in ("x",):
        return X
    return None  # 위 어느 경우에도 안 맞으면 '알 수 없는 라벨'이라는 뜻으로 None.


# =============================================================
# 2. 패턴 생성기 (보너스 과제 2)
#    크기 N 을 받아 N x N 십자가(Cross) / X 패턴을 자동 생성한다.
# =============================================================

def make_cross(n):
    """가운데 행/열이 1인 십자가(Cross) 패턴을 생성한다."""
    c = n // 2   
    # // 는 '몫만 취하는 나눗셈'. 5//2 = 2 (한가운데 칸의 번호).
    # 즉 '지금 칸이 가운데 행(i==c)이거나 가운데 열(j==c)이면 1, 아니면 0'.
    return [[1 if (i == c or j == c) else 0 for j in range(n)] for i in range(n)]


def make_x(n):
    """두 대각선이 1인 X 패턴을 생성한다."""
    # i == j        : 왼쪽 위 -> 오른쪽 아래로 내려가는 대각선
    # i + j == n - 1 : 오른쪽 위 -> 왼쪽 아래로 내려가는 반대쪽 대각선
    # 둘 중 하나라도 참이면 1(대각선 칸), 아니면 0.
    return [[1 if (i == j or i + j == n - 1) else 0 for j in range(n)] for i in range(n)]


# =============================================================
# 3. MAC 연산
# =============================================================

def mac(pattern, filt):
    """
    2차원 배열(패턴)과 필터를 같은 위치끼리 곱하고 모두 더한 값(점수)을 반환한다.
    Multiply(곱하고) + Accumulate(누적해서 더한다) = MAC.
    반복문으로 직접 구현하며 float 결과를 반환할 수 있다.
    """
    n = len(pattern)   # len(리스트): 원소 개수. 2차원 배열이면 '행의 개수' = 한 변의 길이.
    total = 0.0
    for i in range(n):
        prow = pattern[i]     # 입력의 i번째 '행'(가로 한 줄)만 미리 꺼내둠 (반복 안에서 재사용해 조금 빠름)
        frow = filt[i]
        for j in range(n):
            total += prow[j] * frow[j]   # 같은 위치끼리 곱한 값을 total 에 더함.
    return total


def flatten(matrix):
    """2차원 배열을 길이 N^2 의 1차원 배열로 변환한다. (보너스 과제 1)"""
    flat = []
    for row in matrix:
        flat.extend(row)     # .extend(): 리스트의 '원소들을 하나씩' 뒤에 이어붙임.
                             #  (참고: .append(row) 였다면 리스트 자체를 통째로 넣어 2차원이 됨)
    return flat


def mac_1d(pattern_flat, filt_flat):
    """
    1차원으로 펼친 배열에 대한 MAC. 2중 반복/행 인덱싱을 제거해
    메모리 접근 패턴을 단순화한 버전이다. (보너스 과제 1)
    """
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filt_flat[k]
    return total


# =============================================================
# 4. 시간 측정
#    I/O(입출력/파일읽기) 를 제외하고 "연산 함수 호출 구간"만 측정한다.
# =============================================================

def measure(fn, args, repeat):
    """
    fn(*args) 를 repeat 회 반복 호출한 평균 시간(ms)과 마지막 결과를 반환한다.
    - fn   : 실행할 '함수 자체'를 값처럼 받음 (예: mac). 파이썬은 함수도 변수에 담을 수 있음.
    - args : 그 함수에 넘길 인자들을 튜플로 받음 (예: (pattern, filt)).
    """
    result = None
    start = time.perf_counter()   # 현재 시각을 아주 정밀하게 기록 (측정 시작점)
    for _ in range(repeat):       # '_' 는 "반복 횟수만 필요하고 값은 안 쓴다"는 관례적 이름
        result = fn(*args)        # fn(*args): 튜플 args 를 '풀어서' 함수에 각각의 인자로 전달.
                                  #   args=(a, b) 라면 fn(a, b) 와 똑같이 호출됨. (* 가 '풀기' 역할)
    end = time.perf_counter()     # 측정 끝점
    # 걸린 총 시간(초) ÷ 반복 횟수 = 1회 평균(초). ×1000 으로 밀리초(ms) 단위로 변환.
    avg_ms = (end - start) / repeat * 1000.0
    return result, avg_ms


# =============================================================
# 5. 판정 정책 (허용오차 epsilon 기반)
# =============================================================

def judge_ab(score_a, score_b):
    """모드1: 필터 A/B 점수를 비교. 동점(|A-B|<eps)은 '판정 불가'."""
    # abs(x): 절댓값(부호를 뗀 크기). 두 점수 차이가 아주 작으면 '같다'고 판단.
    if abs(score_a - score_b) < EPSILON:
        return "판정 불가"
    return "A" if score_a > score_b else "B" # A 점수가 더 크면 "A", 아니면 "B" 를 반환.


def judge_cross_x(score_cross, score_x):
    """모드2: Cross/X 점수를 비교. 동점(|Cross-X|<eps)은 UNDECIDED."""
    if abs(score_cross - score_x) < EPSILON:
        return UNDECIDED
    return CROSS if score_cross > score_x else X


def fmt(score):
    """점수를 사람이 읽기 좋게 표기. 부동소수점 오차가 있으면 그대로 드러낸다."""
    return str(float(score))


# =============================================================
# 6. 입력 처리 (모드 1) - 행/열 개수, 숫자 파싱 검증 + 재입력 유도
# =============================================================

def read_matrix(n, label):
    """
    콘솔에서 n x n 행렬을 '한 줄씩(공백 구분)' 입력받는다.
    행/열 개수 불일치, 숫자 파싱 실패 시 안내 문구를 출력하고 재입력을 유도한다.
    """
    print(f"{label} ({n}줄 입력, 공백 구분)")
    while True:
        rows = []       # 지금까지 입력받은 행들을 담을 리스트
        ok = True       # 이번 시도가 문제없이 끝났는지 표시하는 깃발(True=정상)
        for _ in range(n):
            line = input().strip()         # input(): 사용자가 친 한 줄을 문자열로 받음
            parts = line.split()           # "1 0 1".split() -> ['1', '0', '1']

            if len(parts) != n:            # 조각 개수가 n 과 다르면(칸 수가 안 맞으면)
                print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 "
                      f"입력하세요. 처음부터 다시 입력하세요.")
                ok = False
                break
            try:
                # [float(p) for p in parts]: 조각(문자열)들을 하나씩 실수로 바꿔 리스트로 만듦.
                # 문자열 "1" -> 숫자 1.0. 만약 "abc" 처럼 숫자가 아니면 ValueError 발생.
                rows.append([float(p) for p in parts])
            except ValueError:             # 위에서 숫자 변환이 실패하면 이리로 옴
                print("입력 형식 오류: 숫자만 입력할 수 있습니다. "
                      "처음부터 다시 입력하세요.")
                ok = False
                break
        # ok 가 True 이고 줄 수도 정확히 n 이면 성공 -> 결과를 돌려주고 while 을 벗어남.
        if ok and len(rows) == n:
            return rows
        # 그렇지 않으면 while True 로 돌아가 처음부터 다시 입력받음.


def run_mode1():
    """모드 1: 필터1 / 필터2 입력 → 사용자 입력(3x3) → MAC → 판정 → 성능(3x3)."""
    n = 3

    print("#" + "-" * 40)
    print("# [1] 필터 입력")
    print("#" + "-" * 40)
    filter_a = read_matrix(n, "필터 A")
    filter_b = read_matrix(n, "필터 B")
    print("  -> 필터 A, B 저장 완료")

    print("#" + "-" * 40)
    print("# [2] 패턴 입력")
    print("#" + "-" * 40)
    pattern = read_matrix(n, "패턴")
    print("  -> 패턴 저장 완료")

    print("#" + "-" * 40)
    print("# [3] MAC 결과")
    print("#" + "-" * 40)
    # measure 가 (결과, 평균시간) 튜플을 돌려주므로, 아래처럼 두 변수로 한 번에 나눠 받습니다.
    # 이것을 '튜플 언패킹(풀어 담기)'이라고 합니다.
    score_a, t_a = measure(mac, (pattern, filter_a), MODE1_REPEAT)
    score_b, t_b = measure(mac, (pattern, filter_b), MODE1_REPEAT)
    avg_ms = (t_a + t_b) / 2.0      # 두 필터 측정 시간의 평균
    verdict = judge_ab(score_a, score_b)

    print(f"A 점수: {fmt(score_a)}")
    print(f"B 점수: {fmt(score_b)}")
    # {avg_ms:.3f} : avg_ms 를 소수점 아래 3자리로 반올림해 표시하라는 '포맷 지정'.
    print(f"연산 시간(평균/{MODE1_REPEAT}회): {avg_ms:.3f} ms")
    if verdict == "판정 불가":
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    else:
        print(f"판정: {verdict}")


# =============================================================
# 7. JSON 로드 / 스키마 검증 (모드 2)
# =============================================================

def load_filters(raw_filters):
    """
    filters(size_5/size_13/size_25) 를 읽어
    { N : { 'Cross': [[...]], 'X': [[...]] } } 형태로 정규화한다.
    내부 키('cross','x')는 표준 라벨로 정규화한다.
    """
    filters_by_size = {} # 키: 사이즈 , 값: { 'Cross': [[...]], 'X': [[...]] }
    loaded = []

    for size_key, label_map in raw_filters.items():
        try:
            n = int(size_key.split("_")[1])
        #   IndexError: [1] 자리가 없을 때 / ValueError: 정수로 못 바꿀 때
        except (IndexError, ValueError):
            continue   # continue: 이번 항목은 건너뛰고 다음 반복으로.

        norm = {} # 키: 표준 이름(Cross/X), 값: 필터 
        for label_key, matrix in label_map.items():
            std = normalize_label(label_key)           # 표준 라벨(Cross/X)로 변환
            if std is not None:
                norm[std] = matrix                     # 표준 이름을 키로 필터 격자를 저장
        filters_by_size[n] = norm

        # "Cross, X 중 실제로 있는 것만 골라 쉼표로 이어붙인 문자열" (로그 출력용).
        have = ", ".join(k for k in (CROSS, X) if k in norm)
        loaded.append((n, have))
    return filters_by_size, loaded


def extract_n_from_pattern_key(key):
    """size_{N}_{idx} 형태의 키에서 N 을 추출한다. 실패 시 None."""
    parts = key.split("_")
    if len(parts) < 3 or parts[0] != "size":   # 조각이 3개 미만이거나 첫 조각이 'size'가 아니면 형식 오류
        return None
    try:
        return int(parts[1])             # 가운데 조각을 숫자로 (여기서는 N)
    except ValueError:
        return None


def pattern_sort_key(key):
    """
    패턴 키를 (N, idx) 숫자 기준으로 정렬하기 위한 키.
    size_5_1 -> (5, 1). 형식이 어긋나는 키는 맨 뒤로 보낸다.
    """
    parts = key.split("_")
    try:
        # (0, N, idx) 처럼 튜플을 반환하면, 정렬은 앞 요소부터 차례로 비교합니다.
        # 맨 앞을 0 으로 둬서 '정상 키'끼리 먼저, 그 안에서 N -> idx 순으로 정렬되게 함.
        return (0, int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (1, 0, 0)   # 형식이 이상한 키는 앞자리를 1 로 둬서 정상 키들보다 뒤로 밀림.


def is_square(matrix, n):
    """matrix 가 n x n 정방 2차원 배열인지 확인한다."""
    # isinstance(x, list): x 가 리스트 타입인지 검사. (숫자나 None 이 잘못 들어온 경우를 걸러냄)
    if not isinstance(matrix, list) or len(matrix) != n:
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != n:   # 각 행도 리스트이고 길이가 n 이어야 함
            return False
    return True


def evaluate_case(key, case, filters_by_size):
    """
    패턴 케이스 하나를 평가한다. 스키마/크기 문제가 있어도 예외로 프로그램이
    중단되지 않도록 케이스 단위로 방어하고, 문제가 있으면 FAIL 로 처리한다.
    반환: dict(key, status, reason, score_cross, score_x, verdict, expected)
    """
    # 결과를 담을 딕셔너리를 미리 '기본값=FAIL'로 만들어 둡니다.
    # 아래에서 검사를 통과할 때마다 값을 채우고, 문제가 생기면 사유를 적고 즉시 return 합니다.
    # 조기 반환(early return): 문제가 있으면 곧바로 반환하는 방식
    res = {
        "key": key, "status": "FAIL", "reason": "",
        "score_cross": None, "score_x": None,
        "verdict": None, "expected": None,
    }

    # (1) 키에서 크기 N 추출
    n = extract_n_from_pattern_key(key)
    if n is None:
        res["reason"] = "키 형식 오류: size_{N}_{idx} 규칙에서 N 추출 실패"
        return res

    # (2) 패턴 input 확인 + 정방 검증
    # case.get("input"): 딕셔너리에서 'input' 값을 꺼내되, 없으면 None 을 돌려줌(오류 안 남).
    # 앞의 isinstance 검사로 case 가 딕셔너리일 때만 .get 을 시도해 안전하게 처리.
    inp = case.get("input") if isinstance(case, dict) else None
    if inp is None:
        res["reason"] = "스키마 오류: 'input' 필드 없음"
        return res
    if not is_square(inp, n):
        res["reason"] = f"크기 불일치: input 이 {n}x{n} 형태가 아님"
        return res

    # (3) 해당 크기의 Cross/X 필터 선택 + 크기 일치 검증
    fmap = filters_by_size.get(n)
    if not fmap:                       # 해당 크기 필터가 아예 없으면(빈 값 포함) 실패 처리
        res["reason"] = f"필터 없음: size_{n} 필터를 찾을 수 없음"
        return res
    
    cross_f = fmap.get(CROSS)
    x_f = fmap.get(X)
    if cross_f is None or x_f is None:
        res["reason"] = f"필터 누락: size_{n} 에 Cross/X 필터가 모두 필요"
        return res
    if not is_square(cross_f, n) or not is_square(x_f, n):
        res["reason"] = f"필터 크기 불일치: size_{n} 필터가 {n}x{n} 아님"
        return res

    # (4) expected 정규화 ('+'/'x' -> Cross/X)
    expected = normalize_label(case.get("expected"))
    if expected is None:
        res["reason"] = f"라벨 오류: 알 수 없는 expected 값 '{case.get('expected')}'"
        return res
    res["expected"] = expected

    # (5) MAC 연산 + 판정
    score_cross = mac(inp, cross_f)    # 입력을 십자가 필터와 겹쳐 점수 계산
    score_x = mac(inp, x_f)            # 입력을 X 필터와 겹쳐 점수 계산
    verdict = judge_cross_x(score_cross, score_x)
    res["score_cross"] = score_cross
    res["score_x"] = score_x
    res["verdict"] = verdict

    # (6) PASS/FAIL : 판정 결과가 정답(expected)과 같으면 통과.
    if verdict == expected:
        res["status"] = "PASS"
    else:
        res["status"] = "FAIL"
        if verdict == UNDECIDED:
            res["reason"] = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
        else:
            res["reason"] = f"판정({verdict}) != expected({expected})"
    return res


def performance_table():
    """
    크기별 MAC 평균 연산 시간(ms)과 연산 횟수(N^2)를 측정해 표로 출력한다.
    보너스 1: 2차원 vs 1차원(flatten) 접근 방식을 함께 비교한다.
    """
    print("#" + "-" * 40)
    print(f"# [3] 성능 분석 (평균/{PERF_REPEAT}회)")
    print("#" + "-" * 40)
    print(f"{'크기':<8}{'2D 시간(ms)':>14}{'1D 시간(ms)':>14}{'연산 횟수(N^2)':>16}")
    print("-" * 52)

    for n in (3, 5, 13, 25):        # 네 가지 크기에 대해 각각 측정
        pattern = make_cross(n)     # 측정용 입력(십자가) 자동 생성
        filt = make_cross(n)
        _, t2d = measure(mac, (pattern, filt), PERF_REPEAT)

        pf = flatten(pattern)       # 1차원 버전 비교용으로 펼치기
        ff = flatten(filt)
        _, t1d = measure(mac_1d, (pf, ff), PERF_REPEAT)

        size_label = f"{n}x{n}"
        # {t2d:>14.4f}: 오른쪽 정렬 14칸 + 소수점 4자리. n * n 이 곧 연산 횟수(N^2).
        print(f"{size_label:<8}{t2d:>14.4f}{t1d:>14.4f}{n * n:>16}")
    print()
    print("  * 2D=중첩 반복 접근, 1D=길이 N^2 배열 접근(보너스1). "
          "시간은 대략 N^2 에 비례해 증가.")


def run_mode2(data_path=DATA_FILE):
    """모드 2: data.json 로드 → 검증/판정 → 성능 분석 → 결과 요약."""
    if not os.path.exists(data_path):
        print(f"오류: '{data_path}' 파일을 찾을 수 없습니다. "
              f"main.py 와 같은 폴더에 두고 다시 실행하세요.")
        return
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"오류: data.json 을 읽는 중 문제가 발생했습니다 -> {e}")
        return

    # [1] 필터 로드
    print("#" + "-" * 40)
    print("# [1] 필터 로드")
    print("#" + "-" * 40)

    filters_by_size, loaded = load_filters(data.get("filters", {}))
    for n, have in sorted(loaded):
        print(f"  size_{n} 필터 로드 완료 ({have})")

    # [2] 패턴 분석 (라벨 정규화 적용)
    print("#" + "-" * 40)
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#" + "-" * 40)
    patterns = data.get("patterns", {})
    results = []                          # 각 케이스의 채점 결과를 모아둘 리스트

    # sorted(..., key=pattern_sort_key): size_5_1, size_5_2, size_13_1
    for key in sorted(patterns.keys(), key=pattern_sort_key):
        res = evaluate_case(key, patterns[key], filters_by_size)
        results.append(res)
        print(f"--- {key} ---")

        if res["score_cross"] is not None:
            print(f"Cross 점수: {fmt(res['score_cross'])}")
            print(f"X 점수: {fmt(res['score_x'])}")

            exp = res["expected"] if res["expected"] else "?"
            # 아래는 문자열을 + 로 이어붙이는데, 뒤쪽은 'FAIL 이고 사유가 있을 때만' 사유를 덧붙임.
            print(f"판정: {res['verdict']} | expected: {exp} | {res['status']}"
                  + (f" ({res['reason']})" if res["status"] == "FAIL" and res["reason"] else ""))
        else:
            # 스키마/크기 문제로 판정 자체가 불가능한 경우(점수가 없음)
            print(f"판정: - | {res['status']} ({res['reason']})")
        print()

    # [3] 성능 분석
    performance_table()
    print()

    # [4] 결과 요약
    total = len(results)                 # 전체 케이스 수
    
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed
    print("#" + "-" * 40)
    print("# [4] 결과 요약")
    print("#" + "-" * 40)
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")
    if failed:                           # 실패가 1개 이상이면(0 은 거짓으로 취급) 목록 출력
        print("실패 케이스:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['key']}: {r['reason']}")


# =============================================================
# 8. 실행 흐름 (메뉴)
# =============================================================

def main():
    print("=== Mini NPU Simulator ===")
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    choice = input("선택: ").strip()

    if choice == "1":
        run_mode1()
    elif choice == "2":
        run_mode2()
    else:
        print("잘못된 선택입니다. 1 또는 2 를 입력하세요.")


if __name__ == "__main__":
    main()
