import execjs

# 读取 JavaScript 文件
with open("svgnest.js", "r", encoding="utf-8") as f:
    js_code = f.read()

# 编译并调用 JavaScript 函数
ctx = execjs.compile(js_code)
result = ctx.call("SvgNest.parsesvg", "<svg>...</svg>")  # 调用 JavaScript 函数
print(result)