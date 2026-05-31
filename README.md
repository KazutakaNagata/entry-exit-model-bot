# swing-bot-v2

BTCJPY の 1分足データを前提に、手数料込みで成立する swing / trend bot を研究するための新規リポジトリです。

この初期状態では、**まだ学習コード・特徴量コード・バックテストコードは実装していません**。まず、AI 実装で起きやすいリークや test 汚染を防ぐため、研究ルール・モデル設計・検証プロトコル・設定テンプレートだけを固定しています。

## 研究の中心思想

- **entry model** は「この episode を始める価値」を予測する。
- **exit model** は「この episode を続ける価値」を予測する。
- 1 episode は、entry から exit までの 1ポジション。
- 往復コストは初期値として **15bps** を想定する。
- 短期リターンの方向予測ではなく、**手数料後 episode P/L** を主役にする。
- test は最終監査専用。調整は valid 5 fold のみで行う。

## 初期モデル案

### Entry

- `LGBMRegressor`
- target: `side-normalized fixed-hold net_return_bps`
- initial horizons: `H = 60m, 120m, 240m`
- initial models:
  - `entry_long_H60`, `entry_long_H120`, `entry_long_H240`
  - `entry_short_H60`, `entry_short_H120`, `entry_short_H240`

### Exit

- `LGBMRegressor`
- target: `hold_delta_bps`
- initial lookaheads:
  - long: `K = 30m, 60m`
  - short: `K = 15m, 30m`

## 重要な時刻ルール

1分足の row `t` に対して:

- features は `t` 以前の確定済み 1分足だけから作る。
- decision は `t` の足が確定した後に行う。
- execution は原則として `t+1 open`。
- entry label は `t+1 open` から `t+1+H open` で作る。
- exit label は `t+1 open` から `t+1+K open` の差分で作る。

`close[t]` で特徴量を作って `close[t]` で約定したことにする実装は禁止です。

## 次にやること

推奨順序は以下です。

1. 1分足データ loader と data audit。
2. train / valid 5 fold / test split manifest。
3. cost / entry label / exit label helper。
4. リークしない最小特徴量 pipeline。
5. `entry_long_H60` の valid 評価。
6. multi-horizon entry。
7. long exit model。
8. episode valid backtest。
9. locked config を作ってから test audit。

詳細は `docs/IMPLEMENTATION_PLAN.md` を参照してください。

## Codex / AI 実装者向け

実装前に必ず次を読んでください。

- `AGENTS.md`
- `docs/RESEARCH_RULES.md`
- `docs/LEAKAGE_RULES.md`
- `docs/VALIDATION_PROTOCOL.md`
- `docs/MODEL_DESIGN.md`
- `docs/REVIEW_CHECKLIST.md`

