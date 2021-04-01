import numpy as np
import numpy.linalg as npla

import warnings

def newton_raphson_poly (p, x_guess, tol=1e-9, max_iter=100):
    # FIXME: Need to account for case in which both function and slope approach zero
    pd = np.polyder(p);
    x = x_guess
    for i in range(max_iter):
        err = x
        x = x - np.polyval(p,x) / np.polyval(pd,x)
        err = abs((x - err) / x)
        print("x, err = ", x, err)
        if err < tol:
            break
    
    return x

def get_exps_lhs(roots, pord):
    
    # Note: Roots must be unique?!

    C = np.zeros([pord, pord], dtype=np.cdouble)
    for i in range(len(roots)):
        pprod = [1]
        for j in range(len(roots)):
            if i != j:
                pprod = np.polymul(pprod, [1, -roots[j]])
        #print(pprod)
        C[0:,i] = pprod[0:]
    exps = npla.solve(C, np.array((pord-1)*[0.]+[1.]))
    
    # # test
    # pprod = [0]
    # for r,e in zip(roots, exps):
    #     pdiv, pdiv_r = np.polydiv(p, [1.0, -r])
    #     pdiv = e * pdiv
    #     pprod = np.polyadd(pprod, pdiv)
    # print("error = ", np.sum(np.abs(pprod-np.array((pord-1)*[0.]+[1.]))))
    
#     print("roots = \n", np.transpose([roots]))
#     print("exponents = \n", np.transpose([exps]))
    
    return exps

def sv_lhs (x, x0, roots, exps):
    
    warnings.simplefilter("ignore")
        
    y = 1.0
    for r,e in zip(roots, exps):
        y *= np.power((x - r)/(x0 - r), e)
        # y *= ((x - r)/(x0 - r)) ** e
    return np.real(y)

def bisection_sv (xm, x1, x2, 
                  m0, roots, exps, 
                  t, alpha,
                  counter=0, tol=1e-10, max_iter=1000):
    
    fm = sv_lhs (xm, m0, roots, exps) - np.exp(- alpha * t)
    f1 = sv_lhs (x1, m0, roots, exps) - np.exp(- alpha * t)
    f2 = sv_lhs (x2, m0, roots, exps) - np.exp(- alpha * t)
    
    if np.isnan(f1):
        if 0 < np.real(sv_lhs (x1+tol, m0, roots, exps) \
                       - np.exp(- alpha * t)):
            f1 = float('inf')
        else:
            f1 = -float('inf')
    
    if np.isnan(f2):
        if 0 < np.real(sv_lhs (x2-tol, m0, roots, exps) \
                       - np.exp(- alpha * t)):
            f2 = float('inf')
        else:
            f2 = -float('inf')
    
    # print("function evals", f1, fm, f2)
    
    if counter > max_iter:
        print("Convergence not reached.")
        #return float('nan')
        return 0.0
    
    err = np.abs((x2-x1)/xm)
    #err = np.abs((x2-x1)/xm)
    if err < tol:
        #print("num iter = ", counter)
        return xm
    
#     print("bisection step (x, err): ", xm, err)
    
    if np.sign(fm)*np.sign(f1) > 0 \
            and np.sign(fm)*np.sign(f2) > 0:
        #print("Error! No zero found.")
        #return float('nan')
        #return 0.0
        return xm
    else:
        if np.sign(fm)*np.sign(f1) > 0:
            x = bisection_sv(
                    0.5*(xm+x2), xm, x2, 
                    m0, roots, exps, t, alpha,
                    counter+1, tol, max_iter)
            return x
        elif np.sign(fm)*np.sign(f2) > 0:
            x = bisection_sv(
                    0.5*(x1+xm), x1, xm,
                    m0, roots, exps, t, alpha,
                    counter+1, tol, max_iter)
            return x
        else:
            #print("Warning! Case not accounted.")
            #print("num iter = ", counter)
            #return float('nan')
            return xm

def zero_finder_sv (x_guess, x0, roots, exps, 
                    t, alpha,
                    tol=1e-10, max_iter=1000):
    
    x = x_guess
    
    try:
        # Set default max bounds
        bound_lo, bound_hi = -1.0e3, 1.0e3

        imag_tol = 1.0e-12
        r_sort = []
        for r in roots:
            if np.abs(np.imag(r)) < imag_tol:
                r_sort.append(np.real(r))
        r_sort.sort()
        if len(r_sort) == 1:
            bound_lo = min(x_guess, r_sort[0])
            bound_hi = max(x_guess, r_sort[0])
        else:
            if x_guess <= r_sort[0]:
                bound_hi = r_sort[0]
            elif x_guess >= r_sort[-1]:
                bound_lo = r_sort[-1]
            else:
                for i in range(1, len(r_sort)):
                    if x_guess <= r_sort[i]:
                        bound_lo, bound_hi = r_sort[i-1], r_sort[i]
                        break

    #     print("Bounds: ", bound_lo, bound_hi)
    #     print("Initial guess: ", x_guess)

        # h = tol * min (bound_hi - x_guess, x_guess - bound_lo)
        h = 0.0
        x = bisection_sv (x_guess, bound_lo+h, bound_hi-h, 
                          x0, roots, exps, t, alpha,
                          tol = tol)
    except Exception() as exp:
        print('Convergence reached.')
        #print(exp)
    
    return x
