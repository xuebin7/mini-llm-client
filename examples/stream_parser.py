from mini_llm.streaming import StreamParser

lines = [
    'data: {"choices":[{"delta":{"content":"Hello"}}]}',
    "",
    'data: {"choices":[{"delta":{"content":" World"}}]}',
    "",
    'data: {"choices":[{"delta":{"content":"!"}}]}',
    "",
    "data: [DONE]",
]

parser = StreamParser()

for text in parser.parse(lines):
    print(text, end="", flush=True)

print()