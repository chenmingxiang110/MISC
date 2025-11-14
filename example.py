def func_add(max_digit=4, is_add=True):
    
    def instruct_level_0(a, b):
        res = "=("
        a_str, b_str = str(a), str(b)
        for i_digit, digit in enumerate(a_str):
            res = res + digit + "*10^" + str(len(a_str) - i_digit - 1) + "+"
        if is_add:
            res = res[:-1] + ")+("
        else:
            res = res[:-1] + ")-("
        for i_digit, digit in enumerate(b_str):
            res = res + digit + "*10^" + str(len(b_str) - i_digit - 1) + "+"
        res = res[:-1] + ")"
        return res
    
    def instruct_level_1(a, b):
        res = "="
        a_str, b_str = str(a), str(b)
        n_digit = max(len(a_str), len(b_str))
        for i_digit in range(n_digit):
            _res = []
            _i_digit = i_digit - n_digit + len(a_str)
            if _i_digit>=0:
                digit = a_str[_i_digit]
                _res.append(digit)
            else:
                _res.append("0")
            _i_digit = i_digit - n_digit + len(b_str)
            if _i_digit>=0:
                digit = b_str[_i_digit]
                _res.append(digit)
            if len(_res)==1:
                _res = _res[0]
            else:
                if is_add:
                    _res = "(" + _res[0] + "+" + _res[1] + ")"
                else:
                    _res = "(" + _res[0] + "-" + _res[1] + ")"
            res = res + _res + "*10^" + str(n_digit - i_digit - 1) + "+"
        res = res[:-1]
        return res
    
    def instruct_level_2(a, b):
        res = "="
        a_str, b_str = str(a), str(b)
        n_digit = max(len(a_str), len(b_str))
        for i_digit in range(n_digit):
            _res = []
            _i_digit = i_digit - n_digit + len(a_str)
            if _i_digit>=0:
                digit = a_str[_i_digit]
                _res.append(digit)
            else:
                _res.append("0")
            _i_digit = i_digit - n_digit + len(b_str)
            if _i_digit>=0:
                digit = b_str[_i_digit]
                _res.append(digit)
            if is_add:
                _res = str(sum([int(x) for x in _res]))
            else:
                _res = _res[0] if len(_res)==1 else str(int(_res[0])-int(_res[1]))
            if _res[0]=='-' and i_digit!=0:
                res = res[:-1] + _res + "*10^" + str(n_digit - i_digit - 1) + "+"
            else:
                res = res + _res + "*10^" + str(n_digit - i_digit - 1) + "+"
        res = res[:-1]
        return res
    
    def instruct_level_3(a, b):
        res = "="
        a_str, b_str = str(a), str(b)
        n_digit = max(len(a_str), len(b_str))
        nums = []
        for i_digit in range(n_digit):
            _res = []
            _i_digit = i_digit - n_digit + len(a_str)
            if _i_digit>=0:
                digit = a_str[_i_digit]
                _res.append(digit)
            else:
                _res.append("0")
            _i_digit = i_digit - n_digit + len(b_str)
            if _i_digit>=0:
                digit = b_str[_i_digit]
                _res.append(digit)
            if is_add:
                _res = sum([int(x) for x in _res])
            else:
                _res = int(_res[0]) if len(_res)==1 else int(_res[0])-int(_res[1])
            num = _res * 10**(n_digit - i_digit - 1)
            if num>=0 or i_digit==0:
                res = res + str(num) + "+"
            else:
                res = res[:-1] + str(num) + "+"
            nums.append(num)
        res = res[:-1]
        while len(nums)>1:
            nums[-2] = nums[-2] + nums[-1]
            nums = nums[:-1]
            res = res + "\n="
            for i_digit, num in enumerate(nums):
                if num>=0 or i_digit==0:
                    res = res + str(num) + "+"
                else:
                    res = res[:-1] + str(num) + "+"
            res = res [:-1]
        return res
    
    def instruct_level_final(a, b):
        res = "The answer is: "+(str(a+b) if is_add else str(a-b))
        return res
    
    a = np.random.randint(10 ** np.random.randint(1,max_digit+1))
    b = np.random.randint(10 ** np.random.randint(1,max_digit+1))
    if is_add:
        q = str(a)+"+"+str(b)
    else:
        q = str(a)+"-"+str(b)
    instruct = "\n".join([
        q, instruct_level_0(a,b), instruct_level_1(a,b),
        instruct_level_2(a,b), instruct_level_3(a,b), instruct_level_final(a,b)
    ])
    return q, instruct

def func_multiply(max_digit=3):
    
    def instruct_level_0(a, b):
        res = "=("
        a_str, b_str = str(a), str(b)
        for i_digit, digit in enumerate(a_str):
            res = res + digit + "*10^" + str(len(a_str) - i_digit - 1) + "+"
        res = res[:-1] + ")*("
        for i_digit, digit in enumerate(b_str):
            res = res + digit + "*10^" + str(len(b_str) - i_digit - 1) + "+"
        res = res[:-1] + ")"
        return res
    
    def instruct_level_1(a, b):
        res = "="
        a_str, b_str = str(a), str(b)
        max_len = max(len(a_str), len(b_str))
        a_str = "0" * (max_len - len(a_str)) + a_str
        b_str = "0" * (max_len - len(b_str)) + b_str
        a_int, b_int = [[int(x) for x in s] for s in [a_str, b_str]]
        for a_digit in range(max_len):
            a_power = max_len - a_digit - 1
            for b_digit in range(max_len):
                b_power = max_len - b_digit - 1
                num_a, num_b = a_int[a_digit], b_int[b_digit]
                if num_a!=0 and num_b!=0:
                    res = res + f"({num_a}*{num_b})*10^({a_power}+{b_power})+"
        res = res[:-1] + "\n="
        for a_digit in range(max_len):
            a_power = max_len - a_digit - 1
            for b_digit in range(max_len):
                b_power = max_len - b_digit - 1
                num_a, num_b = a_int[a_digit], b_int[b_digit]
                if num_a!=0 and num_b!=0:
                    res = res + f"{num_a*num_b}*10^{a_power+b_power}+"
        res = res[:-1]
        nums = []
        for a_digit in range(max_len):
            a_power = max_len - a_digit - 1
            for b_digit in range(max_len):
                b_power = max_len - b_digit - 1
                num_a, num_b = a_int[a_digit], b_int[b_digit]
                if num_a!=0 and num_b!=0:
                    nums.append(num_a*num_b * 10 ** (a_power+b_power))
        while len(nums)>1:
            res = res + "\n="
            for i_digit, num in enumerate(nums):
                if num>=0 or i_digit==0:
                    res = res + str(num) + "+"
                else:
                    res = res[:-1] + str(num) + "+"
            res = res [:-1]
            nums[-2] = nums[-2] + nums[-1]
            nums = nums[:-1]
        res = res + f"\n={nums[0]}"
        return res
    
    def instruct_level_final(a, b):
        res = "The answer is: "+str(a*b)
        return res
    
    a = np.random.randint(10 ** np.random.randint(1,max_digit+1))
    b = np.random.randint(10 ** np.random.randint(1,max_digit+1))
    q = str(a)+"*"+str(b)
    if a<10 and b<10:
        instruct = q + f"\n={a*b}"
    else:
        instruct = "\n".join([
            q, instruct_level_0(a,b), instruct_level_1(a,b), instruct_level_final(a,b)
        ])
    return q, instruct

question, answer = func_multiply()
print(question)
print(answer)
