import numpy as np
import matplotlib.pyplot as plt

ci_0001 = [np.array([ 5.08214870e-20, -1.34567188e-15, -8.66592999e-17,  2.66753120e-16,
       -1.34642464e-15,  3.47781986e-16,  1.82754153e-15,  2.78312877e-16,
        5.30144370e-16, -8.79816925e-02, -8.93070228e-13,  8.81797910e-03,
       -1.14572471e+00,             np.nan, -2.72129710e-01, -4.72998416e-01,
       -9.21235215e-02, -2.24627419e-01, -1.21352771e-01,  3.71637866e-01,
        3.12527624e-01, -3.65975663e-01, -3.44870998e-01,  1.23253446e+00,
        1.28996411e-01, -3.72971306e-01, -4.59329701e-02, -5.59717576e-01,
       -4.79452038e-01,  1.92265052e-02,  3.14349219e-01, -3.25536202e-01,
        4.36944906e-02,  3.22944072e-02,  2.71009009e-02, -2.32677158e-02,
        6.87990431e-02, -3.04722220e-01, -1.63732565e-01,  1.90765959e-01,
        5.16395828e-01, -4.88641542e-01,  6.71413101e-03,  7.61646718e-01,
        2.49261666e-01, -1.22104956e-01,  6.98409409e-02, -4.69226804e-01,
       -2.96071410e-02,  2.24196900e-01,  5.16395828e-01, -3.69528430e-01,
       -6.13213432e-02, -2.69441112e-01,  3.51305708e-01, -7.32335420e-02,
        1.73317954e-01, -2.04751839e-01, -1.16632463e-01,  3.85279467e-01,
        5.15988338e-01, -7.92155836e-02, -9.96862578e-02, -9.37588480e-02,
        4.06400940e-01, -4.11403124e-02,  3.86104297e-01, -6.21425285e-02,
       -6.98490432e-02,  5.04245517e-01]), np.array([ 5.08214870e-20, -1.34567188e-15, -8.66592999e-17,  2.66753120e-16,
       -1.34642464e-15,  3.47781986e-16,  1.82754153e-15,  2.78312877e-16,
        5.30144370e-16,  5.08158085e-01,  8.96485535e-13,  1.20690964e+00,
       -7.02632642e-01,             np.nan,  1.60225880e-01,  3.17429887e-01,
        6.24203743e-01,  2.84317478e-01,  5.80102750e-01,  1.01155298e+00,
        3.98498935e-01,  4.43253334e-01, -3.68454979e-02,  1.41927629e+00,
        7.86246262e-01,  1.08262736e-01,  4.01977318e-01,  1.83912694e-01,
        3.18767825e-02,  2.61964149e-01,  4.07389930e-01,  1.32953487e-01,
        4.95288506e-01,  3.27583632e-01,  4.60513295e-01,  3.70666431e-01,
        6.40020088e-01,  1.18608950e-01,  4.18379231e-01,  7.78206668e-01,
        6.65408393e-01,  3.06854838e-02,  2.69441112e-01,  9.24655352e-01,
        5.99295198e-01,  1.75685829e-01,  5.38189018e-01, -7.83335894e-02,
        4.53932421e-01,  3.99201592e-01,  6.65408393e-01,  1.37361428e-01,
        9.27858751e-02, -6.71413101e-03,  4.69772488e-01,  1.66428426e-02,
        4.77559207e-01,  3.14060122e-02,  5.66328848e-02,  6.64096364e-01,
        6.58551946e-01,  2.20508114e-01,  1.47051908e-01,  5.73143346e-02,
        5.66876141e-01,  5.21220960e-02,  5.11072647e-01,  3.76579728e-02,
        2.79495816e-02,  6.57657686e-01])]

ci_001 = [np.array([-2.09358377e-21, -1.67725087e-15,  1.45999810e-15, -5.26105235e-15,
       -1.39055485e-15, -4.39076228e-15, -2.47227622e-15,  2.16274732e-15,
       -4.03320545e-15, -1.25529711e+00, -2.06945241e-13, -4.53262085e+00,
       -2.77732290e+00, -9.68483704e-14, -3.29368486e-01,  1.17046865e+00,
       -1.12213354e+00, -9.02594807e-01, -4.99911081e-01, -1.35705472e+00,
        6.47576399e-01, -4.20376925e-01,  5.37749702e-01,  1.60471831e+00,
       -8.53882311e-01,  4.76344675e-02, -8.12590101e-01, -9.51762106e-01,
        7.95440686e-01,  8.99294334e-02,  6.38878332e-01, -8.61714732e-01,
        9.69395643e-01, -7.78248300e-01, -4.27022111e-01,  3.75182633e-01,
       -3.51806016e-01, -3.37112499e-01,  2.25135723e-01, -9.22291241e-01,
        9.38250563e-01, -4.00196629e-01, -2.31031448e-01,  1.53464155e-01,
       -9.53280416e-01, -1.64811725e-01, -3.85298316e-01,  1.71154637e-01,
       -6.78980719e-02,  6.10114618e-01,  9.38250563e-01,  5.16136845e-01,
       -2.52118289e-01,  9.61741541e-03,  1.43022514e-01, -5.47276960e-02,
       -4.66204957e-01,  3.16892304e-01,  1.54992031e-02, -2.53764436e-01,
        9.99690403e-01, -1.69954536e-02, -1.99937152e-01,  9.94848468e-02,
       -1.61833161e-01, -1.85021367e-02,  7.15581994e-02,  9.65201028e-02,
        1.11545805e-01,  9.99615346e-02]), np.array([-2.09358377e-21, -1.67725087e-15,  1.45999810e-15, -5.26105235e-15,
       -1.39055485e-15, -4.39076228e-15, -2.47227622e-15,  2.16274732e-15,
       -4.03320545e-15, -7.64976314e-01,  2.09341867e-13, -3.51274214e+00,
       -2.48596107e+00,  9.81991845e-14, -6.58360267e-02,  1.91036264e+00,
       -5.02935368e-01, -5.98234741e-01, -9.43621059e-02, -1.12804203e+00,
        7.28005974e-01,  4.40149389e-01,  7.92186350e-01,  1.83821527e+00,
       -6.32259549e-01,  2.90392338e-01, -5.81569299e-01, -6.19223939e-01,
        1.01211166e+00,  2.49248540e-01,  7.31154803e-01, -3.83263597e-01,
        1.22282570e+00, -5.23800541e-01, -2.32176014e-01,  5.87651177e-01,
       -1.35959746e-01, -1.43349009e-01,  3.98228957e-01, -7.51443347e-01,
        1.04809222e+00,  1.34122144e-01, -9.61741541e-03,  3.26498657e-01,
       -7.97094017e-01,  1.62166345e-02, -1.97259963e-01,  3.87738252e-01,
        1.38075943e-01,  7.26919004e-01,  1.04809222e+00,  8.75606614e-01,
       -9.90301936e-02,  2.31031448e-01,  2.27558381e-01,  3.16255411e-02,
       -3.17716750e-01,  4.15126217e-01,  1.40857781e-01, -1.53063598e-01,
        1.09990464e+00,  2.20637482e-01,  3.74931243e-02,  2.49630646e-01,
       -9.06897733e-02,  6.20705511e-02,  1.44717596e-01,  1.62761535e-01,
        1.89792557e-01,  2.03918616e-01])]

ci_01 = [np.array([-3.57790664e+06,  8.10696198e+10, -4.58377578e+10,  1.93234873e+11,
        1.26307126e+11, -1.21819144e+10, -4.16635610e+10,  1.11710173e+11,
        7.06274825e+10,  3.47642754e+10, -2.29697982e+09, -2.28568786e+01,
       -3.08047233e+11,             np.nan,  1.81123486e+11,  4.04499640e-02,
       -3.47642754e+10, -2.61946855e-01, -1.62261196e-01,  2.15887761e+11,
       -7.13520890e+11, -1.72851570e+11,  4.98576895e+12, -7.44648525e+11,
        3.38880988e+10, -4.11325938e+11,  1.02503844e+11, -4.85889682e+00,
       -3.89965900e-02, -9.71362539e+10,  6.34134506e+11,  1.20535473e+11,
       -6.31393255e+10,  4.98576895e+12,  1.04912316e+11, -5.20899435e+09,
        7.29318228e+10, -1.11202938e-01,  3.58702079e-02,  1.20921403e+10,
        7.22628468e+11,  1.24683734e-01, -2.52695699e+10, -4.97898515e+11,
        3.54316898e+10, -1.06847150e-02,  1.44911212e+11,  1.11198577e-01,
       -2.32183532e-01, -1.09479522e+11,  7.30994290e+11, -6.52368988e+08,
        1.25569774e+12, -2.52695699e+10, -2.19171917e+11, -1.10501401e+11,
        1.09422114e+11,  5.03360430e-02, -5.83416541e-02, -3.28708847e+11,
        7.41328395e+09,  6.91716595e-03, -1.12017930e-01,  1.25569774e+12,
        3.28651439e+11, -2.02900874e-02,  3.28651439e+11,  2.35582701e-02,
        4.73466409e-02, -1.00526823e-01]), np.array([-3.57790664e+06,  8.10696198e+10, -4.58377578e+10,  1.93234873e+11,
        1.26307126e+11, -1.21819144e+10, -4.16635610e+10,  1.11710173e+11,
        7.06274825e+10,  3.47642754e+10, -2.29697982e+09, -1.70261419e+01,
       -3.08047233e+11,             np.nan,  1.81123486e+11,  7.42239523e-01,
       -3.47642754e+10, -2.22896778e-02,  1.49360428e-01,  2.15887761e+11,
       -7.13520890e+11, -1.72851570e+11,  4.98576895e+12, -7.44648525e+11,
        3.38880988e+10, -4.11325938e+11,  1.02503844e+11, -3.48659693e+00,
        7.40510281e-01, -9.71362539e+10,  6.34134506e+11,  1.20535473e+11,
       -6.31393255e+10,  4.98576895e+12,  1.04912316e+11, -5.20899435e+09,
        7.29318228e+10,  4.45399270e-02,  1.67679993e-01,  1.20921403e+10,
        7.22628468e+11,  7.64534099e-01, -2.52695699e+10, -4.97898515e+11,
        3.54316898e+10,  1.41153806e-01,  1.44911212e+11,  3.86803943e-01,
        5.53112082e-02, -1.09479522e+11,  7.30994290e+11, -6.52368987e+08,
        1.25569774e+12, -2.52695699e+10, -2.19171917e+11, -1.10501401e+11,
        1.09422114e+11,  1.46344459e-01,  5.39128582e-02, -3.28708847e+11,
        7.41328395e+09,  2.42120352e-01,  1.23660745e-01,  1.25569774e+12,
        3.28651439e+11,  6.90866518e-02,  3.28651439e+11,  9.55027506e-02,
        1.20982239e-01,  2.52275465e-03])]

ci_1 = [np.array([ 3.87433343e+04, -1.43149118e+11,  4.04077409e+10, -6.21839519e+10,
        3.16524155e+10, -2.04865619e+10, -3.48989513e+10, -1.48687674e+09,
       -1.43008575e+10, -1.97103563e+10,             np.nan, -3.86750429e+02,
       -2.55190115e+10,             np.nan, -5.80339218e+09, -2.95025370e-02,
        1.97103563e+10, -6.65186073e-02, -5.02956082e-02, -2.55137485e+10,
       -1.18329243e+09,  2.32205083e+09, -6.92321407e+09, -1.62509319e+10,
       -1.28864337e+11,  5.71165158e+08,  3.89046197e+10, -5.08216084e+01,
       -7.81284321e+00, -1.67385818e+11,  8.52500376e+09, -3.02536380e+09,
       -1.87625240e+11, -6.92321405e+09, -5.29821723e+11, -1.54790823e+10,
        1.67826980e+11, -3.93942083e-03, -2.71969352e-02, -6.97149518e+11,
        3.20188726e+10, -9.12242426e-02,  2.24767566e+11,  2.01706933e+11,
        2.09768742e+11, -1.20963744e-02,  5.47895556e+11, -3.53524950e-02,
       -2.77635027e-02, -3.38126814e+11, -1.39118689e+10,  4.82047104e+09,
       -2.91374848e+09,  2.24767566e+11, -7.46386335e+11,  2.56409866e+08,
        3.38551016e+11, -1.97953699e-02, -1.93715036e-02, -1.08408895e+12,
       -5.47780800e+10, -5.97512645e-02, -4.27465063e-03, -2.91374848e+09,
        1.08451315e+12, -2.37077735e-03,  1.08451315e+12, -8.91126690e-03,
       -2.10855589e-02, -3.43162453e-02]), np.array([ 3.87433343e+04, -1.43149118e+11,  4.04077409e+10, -6.21839519e+10,
        3.16524155e+10, -2.04865619e+10, -3.48989513e+10, -1.48687674e+09,
       -1.43008575e+10, -1.97103563e+10,             np.nan, -3.58539392e+02,
       -2.55190115e+10,             np.nan, -5.80339218e+09,  5.38927923e-02,
        1.97103563e+10, -5.33889544e-03,  1.30600044e-02, -2.55137485e+10,
       -1.18329242e+09,  2.32205085e+09, -6.92321405e+09, -1.62509319e+10,
       -1.28864337e+11,  5.71165159e+08,  3.89046197e+10, -4.21118674e+01,
       -3.58935104e-01, -1.67385818e+11,  8.52500377e+09, -3.02536380e+09,
       -1.87625240e+11, -6.92321404e+09, -5.29821723e+11, -1.54790823e+10,
        1.67826980e+11,  2.69794764e-02,  4.61148057e-03, -6.97149518e+11,
        3.20188726e+10,  1.97718310e-02,  2.24767566e+11,  2.01706933e+11,
        2.09768742e+11,  1.93630568e-02,  5.47895556e+11,  1.39876930e-02,
        2.17358186e-02, -3.38126814e+11, -1.39118689e+10,  4.82047104e+09,
       -2.91374848e+09,  2.24767566e+11, -7.46386335e+11,  2.56409866e+08,
        3.38551016e+11,  5.38012127e-03,  7.14080851e-03, -1.08408895e+12,
       -5.47780800e+10, -1.40003839e-02,  3.78191135e-02, -2.91374848e+09,
        1.08451315e+12,  2.07431394e-02,  1.08451315e+12,  1.27296927e-02,
        8.55076629e-04, -2.66270845e-03])]


# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot confidence intervals for each noise level for key parameters
colors = ['blue', 'green', 'orange', 'red']
for i, ci in enumerate([ci_0001, ci_001, ci_01, ci_1]):
    lower_vals = ci[0]
    upper_vals = ci[1]
    mid_vals = (lower_vals + upper_vals) / 2
    yerr = (upper_vals - lower_vals) / 2
    x_vals = np.arange(len(mid_vals)) + i * 0.1  # Shift x values slightly for separation
    ax.errorbar(x_vals, mid_vals, yerr=yerr, fmt='o', capsize=5, label=f'Noise: {10**(-len(ci)+i-1) if i != 0 else 0.0001}', color=colors[i])

# Set labels, title, and legend
ax.set_xlabel("Parameter Index")
ax.set_ylabel("Parameter Value")
ax.legend()
ax.set_title("Confidence Intervals for Key Parameters")
plt.show()
