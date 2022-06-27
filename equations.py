import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def Det(a, b, c, d):
    return a * d - b * c

def SystemOfEquations(y, t, m1, m2, r, R, l, g):

    ### y: 0 - theta; 1 - phi; 2 - theta`; 3 - phi`
    yt = np.zeros_like(y)

    ### yt: 0 - theta`; 1 - phi`; 2 - theta``; 3 - phi``

    yt[0] = y[2]
    yt[1] = y[3]

    a11 = (1.5 * m1 + m2) * (R-r)
    a12 = m2 * l * math.cos(y[1] - y[0])
    b1 = -(m1 + m2) * g * math.sin(y[0]) + m2 * l * y[3] ** 2 * math.sin(y[1] - y[0])

    a21 = (R-r) * math.cos(y[1] - y[0])
    a22 = l
    b2 = -g * math.sin(y[1]) - (R-r) * y[2]**2 * math.sin(y[1] - y[0])

    yt[3] = (a11 * b2 - b1 * a21) / (a11 * a22 - a12 * a21)
    yt[2] = -(a12 * b2 - b1 * a22) / (a11 * a22 - a12 * a21)

    return yt


global m1, m2, r, R, l, g

m1 = 2
m2 = 1

R = 0.5
r = 0.1
l = 0.5
g = 10

theta0 = -math.pi / 2
phi0 = 0
dtheta0 = 0
dphi0 = 0

y0 = [theta0, phi0, dtheta0, dphi0]
Tfin = 15
Tstep = 1001
t = np.linspace(0, Tfin, Tstep)



steps = Tstep
OX = 10
OY = 10
mradius = R - r
radius = l
cradius = r
bradius = mradius + cradius



psi = np.linspace(1.5, 4.7, 40)
alpha = np.linspace(0, 6.28, 40)

Y = odeint(SystemOfEquations, y0, t, (m1, m2, r, R, l, g))

tau = Y[:, 0]
phi = Y[:, 1]



O1X = OX - mradius * np.sin(tau)
O1Y = OY - mradius * np.cos(tau)

XCyrcle = bradius * np.sin(psi)
YCyrcle = bradius * np.cos(psi)

X1Cyrcle = cradius * np.sin(alpha)
Y1Cyrcle = cradius * np.cos(alpha)

AX = O1X - radius * np.sin(phi)
AY = O1Y - radius * np.cos(phi)


fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
ax.set(xlim=[6, 11], ylim=[6, 11])


Point_O = ax.plot(OX, OY, marker='o')[0] ### points
Point_O1 = ax.plot(O1X[0], O1Y[0], marker='o')[0]
Point_A = ax.plot(AX[0], AY[0], marker='o')[0]
Drawed_CyrcleB = ax.plot(OX + XCyrcle, OY + YCyrcle)[0]
Drawed_CyrcleM = ax.plot(O1X[0] + X1Cyrcle, O1Y[0] + Y1Cyrcle)[0]

###OO1line = ax.plot([OX, O1X[0]], [OY, O1Y[0]], color=[0, 0, 0])[0]###lines
O1Aline = ax.plot([O1X[0], AX[0]], [O1Y[0], AY[0]], color=[0, 0, 0])[0]

def Kino(i):
    Point_O1.set_data(O1X[i], O1Y[i])
    Point_A.set_data(AX[i], AY[i])
    Drawed_CyrcleB.set_data(OX + XCyrcle, OY + YCyrcle)
    Drawed_CyrcleM.set_data(O1X[i] + X1Cyrcle, O1Y[i] + Y1Cyrcle)
    ###OO1line.set_data([OX, O1X[i]], [OY, O1Y[i]])
    O1Aline.set_data([O1X[i], AX[i]], [O1Y[i], AY[i]])

    return [Point_O1, Point_A, Drawed_CyrcleB, Drawed_CyrcleM, O1Aline]

anima = FuncAnimation(fig, Kino, frames = steps, interval = 10)

plt.show()

plt1.plot(t, Y[:,2], t, Y[:,3])
plt1.xlabel("t")
plt1.ylabel("f(t)")
plt1.legend(["theta", "phi"])
plt1.grid()
plt1.show()

plt2.plot(t, Y[:,2], t, Y[:,3])
plt2.xlabel("t")
plt2.ylabel("f'(t)")
plt2.legend(["theta'", "phi'"])
plt2.grid()
plt2.show()
