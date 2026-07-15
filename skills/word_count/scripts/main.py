"""word_count 技能入口：stdin 读 JSON 参数，stdout 输出统计结果。

沙盒协议：参数是单个 JSON 对象；正常结果打印到 stdout 并以退出码 0 结束，
出错时把原因写到 stderr 并以非 0 退出。
"""

import json
import re
import sys


def main() -> int:
    try:
        args = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        print(f"参数不是合法 JSON: {exc}", file=sys.stderr)
        return 1
    text = str(args.get("text", ""))
    if not text:
        print("缺少 text 参数", file=sys.stderr)
        return 1

    total_chars = len(text)
    non_space_chars = len(re.sub(r"\s", "", text))
    # 中文按单字计，连续的拉丁字母/数字串按一个词计
    words = len(re.findall(r"[一-鿿]|[A-Za-z0-9]+", text))
    lines = len(text.splitlines()) or 1
    print(f"字符数 {total_chars}（去空白 {non_space_chars}），词数 {words}，行数 {lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
