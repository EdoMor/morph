def average(data):
    if not data:
        return 0.0
    else:
        return sum(data) / len(data)