readme

def num2sci(num, n_digit=3):
    power = 0
    while num>10:
        num /= 10
        power += 1
    while num<1:
        num *= 10
        power -= 1
    return str(np.round(num, n_digit))+"e"+str(power)
