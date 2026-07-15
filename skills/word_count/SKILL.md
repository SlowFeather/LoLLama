---
name: word_count
description: 统计一段文本的字符数、词数和行数
label: 数一下字数
entry: scripts/main.py
timeout_sec: 10
parameters:
  text:
    type: string
    description: 要统计的文本内容
required: [text]
---

把用户给出的文本交给本技能统计。输出包含：总字符数（含空白）、
非空白字符数、词数（中文按字、英文按空格分词的粗略口径）和行数。
只在用户明确要求统计字数/词数/行数时使用。
