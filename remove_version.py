import re

with open("requirements.txt") as f:
    data = f.read()

data = re.sub(r'[^a-zA-Z0-9\n.=]', '', data)
data = data.splitlines()
data = [*filter(lambda x: x, data)]
data = [row.split("==")[0] for row in data]
data = [row + "\n" for row in data]

with open("requirements.txt", "w+") as f:
    f.writelines(data)