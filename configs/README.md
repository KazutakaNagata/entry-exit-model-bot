# configs

研究条件を固定する設定ファイルを置きます。

重要:

- 研究途中で既存 config を上書きしない。
- 条件を変える場合は `_v2`, `_v3` のように新規ファイルを作る。
- test に関わる設定変更は locked config 作成後に行わない。

Subdirectories:

```text
costs/    手数料・スリッページ前提
splits/   train / valid / test 分割
features/ feature family / feature set
labels/   entry / exit target 設定
models/   LGBM 設定
policies/ episode policy 設定
```

