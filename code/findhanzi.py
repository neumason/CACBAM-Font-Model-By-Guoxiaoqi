import json

DEFAULT_CHARSET = "./charset/txt9169.json"
cjk = json.load(open(DEFAULT_CHARSET))
CN_CHARSET = cjk["gbk"]



print(CN_CHARSET)
strs = ""
k = 0
for i in CN_CHARSET:
    k+=1
    strs += i
    # if k == 800:
    #     break
print(strs)