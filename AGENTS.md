# AGENTS.md

このリポジトリで AI / Codex が作業するときの最重要ルールです。

## 絶対に守ること

1. **test は最終評価専用**です。通常の train / valid / feature selection / threshold tuning / policy tuning / model selection で test を読んではいけません。
2. 調整は valid 5 fold のみで行ってください。mean だけでなく worst fold を必ず見ます。
3. feature row at time `t` は、`t` 以前に確定済みの 1分足だけから作ってください。
4. entry の約定価格は原則 `t+1 open` です。`close[t]` で特徴量を作って `close[t]` で約定したことにする実装は禁止です。
5. target / label 生成は未来を見てよいが、target 由来の列や未来情報を feature matrix に混ぜてはいけません。
6. 二段目モデルや exit dataset に entry score を使う場合、**OOF prediction のみ使用可**です。in-sample prediction を使ってはいけません。
7. 新しい外部依存は追加しないでください。必要がある場合は、理由を明記して人間レビューを待ってください。
8. 大きな抽象化、複雑なクラス階層、plugin framework を作らないでください。関数中心で、小さくレビューできる差分にしてください。
9. 指示されたタスク範囲を超えて、ついでに学習器・バックテスト・特徴量を増やさないでください。
10. 仕様が曖昧な場合は、勝手に高機能化せず、最小実装に留めて作業メモに不確実性を書いてください。

## 実装前チェック

作業前に以下を確認してください。

- 今回のタスクは docs / data / split / labels / features / training / backtest のどれか。
- test データに触れる必要があるか。通常はありません。
- 未来の high/low/close を feature に使っていないか。
- 全期間fitの scaler / encoder / PCA / feature compression をしていないか。
- fold 分割前に target-aware な処理をしていないか。
- 出力先が `outputs/valid/` か `outputs/test_audit/` か。

## 完了時チェック

可能な範囲で以下を実行・報告してください。

```bash
python -m compileall swing_bot scripts
```

テストが実装されているフェーズでは:

```bash
python -m pytest tests
```

重い学習や本番 backtest は、人間が明示的に依頼するまで実行しないでください。

