"""
Edwards-Anderson order parameter:  
basic numerical technique to solve
the self consistency integral equation.

https://lewiscoleblog.com/spin-glass-models-2

"""

from scipy.integrate import quad

def integrand(x, c):
    return np.exp(-x**2/2)*np.cosh(c*x)**(-2)

n_approx = 100
beta_min = 0
beta_max = 2
beta_array = np.arange(n_approx + 1)*(beta_max - beta_min)/n_approx + beta_min
q_array = np.zeros(n_approx+1)

thresh = 0.001
n_max = 100

for i in range(n_approx+1):
    beta_tmp = beta_array[i]
    q_old = 0
    q_tmp = 1
    j = 0
    while np.abs(q_old - q_tmp) > thresh and j < n_max:
        q_old = q_tmp
        c = beta_tmp*s*np.sqrt(q_old)
        I = quad(integrand, -np.inf, np.inf, args=(c))
        q_tmp = 1 - I[0] / (np.sqrt(2*np.pi))
        j =+ 1
    q_array[i] = q_tmp

plt.plot(beta_array, q_array)
plt.xlabel(r"$\beta$")
plt.ylabel("q")
plt.title("Edwards-Anderson Order Parameter (s=1)")
plt.show()
