# Functions research using a sympy library
# Theme: Math
from sympy import *
import numpy as np

def f_research(*args, xlim=None, ylim=None, alternative=False, steps=np.linspace(-10, 10, 1001).round(7)):
    
    x, f = args[0], args[1]
    
    plot(*args[1:], axis_center=(0, 0), xlim=xlim, ylim=ylim)
    
    print('\nФункция:')
    display(Eq(Function('f')(x), f))
    
    print('\nПроизводная функции:')
    display(Eq(Function('f\'')(x), f.diff()))
    
    # 1 even and odd functions definition
    if f == f.subs(x, -x):
        print('\n1) Функция четная')
    elif -f == f.subs(x, -x):
        print('\n1) Функция нечетная')
    else:
        print('\n1) Функция ни четная, ни нечетная')
        
    # 2 zeros of a function definition
    print('\n2) Нули функции получаются при точных значениях x:')
    # if there is no roots, it won't be able to apply evalf() function. Also if the values is a complex number, it forces the error too. 
    # That's why i'm using exceptions.
    try:
        display(solveset(f, x).evalf())
    except (ValueError, AttributeError):
        pass
    
    # 3 positive values definition
    print('\n3) Значения функции принимают положительные значения на промежутках x:')
    try:
        display(solveset(f>0, x, domain=Interval(-oo, oo)).evalf())
    except (ValueError, AttributeError):
        pass
    
    # 4 negative values definition
    print('\n4) Значения функции принимают отрицательные значения на промежутках x:')
    try:
        display(solveset(f<0, x, domain=Interval(-oo, oo)).evalf())
    except (ValueError, AttributeError):
        pass
    
    # 5 increasing and decreasing intervals definition
    if alternative is False:
        intervals_ = intervals_of_constancy(x, f)
    else:
        intervals_ = intervals_of_constancy_alt(x, f, steps=steps)
    print('\n5) Функция возрастает на промежутках x:')
    try:
        display(intervals_[0].evalf())
    except (ValueError, AttributeError):
        pass
    print('\n Функция убывает на промежутках x:')
    try:
        display(intervals_[1].evalf())
    except (ValueError, AttributeError):
        pass
        
      
def intervals_of_constancy(*args):
    
    x, f = args[0], args[1]
    
    # RANGE OF VALID VALUES DEFINITION
    valid_range = solveset(f > -oo, x, domain=Interval(-oo, +oo))
    
    # compute derivative of function and takes positive values as intervals of increasing of function
    try:
        increase_intervals = solveset(f.diff() > 0, x, domain=valid_range)
    except ValueError:
        increase_intervals = None
        
    # compute derivative of function and takes positive values as intervals of decreasing of function
    try:
        decrease_intervals = solveset(f.diff() < 0, x, domain=valid_range)
    except ValueError:
        decrease_intervals = None
        
    # returns intervals of increasing and decreasing values 
    return increase_intervals, decrease_intervals


def intervals_of_constancy_alt(*args, steps=np.linspace(-10, 10, 1001).round(7)):
    
    x, f = args[0], args[1]
    
    # compute function values for discrete steps
    func_steps = [f.subs(x, i) for i in steps]
    
    # intervals of increasing values
    increase_intervals = []
    # intervals of decreasing values
    decrease_intervals = []
    
    # check every step of function
    for i in range(1, len(func_steps)-1):
        
        # 2 points around current point are bigger
        if func_steps[i-1] > func_steps[i] < func_steps[i+1]:

            if len(increase_intervals) == 0:
                last_interval = Interval(steps[0], steps[i])
                decrease_intervals.append(last_interval)

            else:
                last_interval = Interval(last_interval.right, steps[i])
                decrease_intervals.append(last_interval)
                
        # 2 points around current point are fewer
        elif func_steps[i-1] < func_steps[i] > func_steps[i+1]:

            if len(decrease_intervals) == 0:
                last_interval = Interval(steps[0], steps[i])
                increase_intervals.append(last_interval)
            else:
                last_interval = Interval(last_interval.right, steps[i])
                increase_intervals.append(last_interval)
                
                
    # computing range of valid values and append it in the necessary places
    else:
        # RANGE OF VALID VALUES DEFINITION
        valid_range = solveset(f > -oo, x, domain=Interval(-oo, +oo))
        
        # periodic functions
        if len(increase_intervals) > 0 and len(decrease_intervals) > 0:

            if increase_intervals[-1].right > decrease_intervals[-1].right:
                decrease_intervals.append(Interval(last_interval.right, steps[i+1]))
            else:
                increase_intervals.append(Interval(last_interval.right, steps[i+1]))

        # not periodic functions, 1 extremum
        elif len(increase_intervals) > 0 and len(decrease_intervals) == 0:
            decrease_intervals = Interval.open(last_interval.right, valid_range.right)
            increase_intervals = Interval.open(valid_range.left, last_interval.right)
            
        elif len(increase_intervals) == 0 and len(decrease_intervals) > 0:
            increase_intervals = Interval.open(last_interval.right, valid_range.right)
            decrease_intervals = Interval.open(valid_range.left, last_interval.right)
            
        # no extremums
        else:
            if func_steps[-1] > func_steps[0]:
                increase_intervals = Interval.open(valid_range.left, valid_range.right)
                
            else:
                decrease_intervals = Interval.open(valid_range.left, valid_range.right)
                
    # returns intervals of increasing and decreasing values
    return increase_intervals, decrease_intervals   

# solve MSE
def MSE_solve(*args):
    
    x, f, points = args[0], args[1], args[2]
    f_solved_list = np.array([])
    misses = np.array([])
    
    for x_cord, y_cord in points:
        f_solved = f.subs(x, x_cord)
        miss = f.subs(x, x_cord).evalf() - y_cord
        misses = np.append(misses, miss)
        f_solved_list = np.append(f_solved_list, f_solved)
        
    MSE = (misses**2).sum()/len(misses)
    RMSE = sqrt(MSE).evalf()
    
    return MSE, RMSE

# get a function by interpolation
def resurrected(f, coefs, points):
    eq_main = []
    for token in points:
        eq_main.append(Eq(f, token[1]).subs(x, token[0]))

    coefs_solved = nonlinsolve(eq_main, coefs).args[0]

    for i in range(len(coefs_solved)):
        f = (f).subs(coefs[i], coefs_solved[i])
        
    return f