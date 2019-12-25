import numpy as np

jaw_mask = np.array(
    [356, 372, 373, 375, 462, 463, 496, 497, 563, 564, 649, 650, 784, 1210, 1213, 1359, 1360, 1386, 1575, 1576, 1577,
     1578, 1579, 1584, 1585, 1586, 1587, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1657, 1658, 1667, 1668, 1669,
     1670, 1686, 1687, 1691, 1693, 1694, 1695, 1696, 1697, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708,
     1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1726, 1733, 1734, 1735, 1736, 1737, 1738, 1740, 1749, 1750,
     1751, 1758, 1759, 1763, 1766, 1767, 1768, 1769, 1770, 1771, 1773, 1774, 1775, 1776, 1778, 1787, 1788, 1789, 1790,
     1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1826, 1827, 1836, 1848, 1849,
     1850, 1863, 1864, 1865, 1866, 1940, 1941, 2072, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2092, 2100, 2149, 2150,
     2531, 2712, 2713, 2714, 2715, 2720, 2721, 2722, 2723, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2774, 2775,
     2784, 2785, 2786, 2787, 2803, 2804, 2808, 2810, 2811, 2812, 2813, 2814, 2817, 2818, 2819, 2820, 2821, 2822, 2823,
     2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2836, 2837, 2838, 2845, 2846, 2847, 2848, 2849,
     2850, 2851, 2852, 2853, 2855, 2863, 2864, 2865, 2866, 2869, 2870, 2871, 2874, 2875, 2876, 2877, 2878, 2879, 2880,
     2881, 2882, 2883, 2885, 2890, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905,
     2906, 2907, 2928, 2929, 2934, 2937, 2938, 2939, 2946, 2947, 2948, 2949, 3107, 3109, 3112, 3113, 3114, 3115, 3116,
     3118, 3119, 3188, 3189, 3190, 3191, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3220, 3221, 3243,
     3255, 3256, 3257, 3258, 3263, 3264, 3265, 3268, 3269, 3270, 3271, 3276, 3277, 3278, 3281, 3283, 3284, 3285, 3286,
     3287, 3288, 3289, 3291, 3292, 3293, 3294, 3305, 3307, 3308, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3323,
     3324, 3344, 3353, 3354, 3355, 3356, 3357, 3362, 3363, 3364, 3367, 3368, 3369, 3370, 3376, 3377, 3378, 3379, 3389,
     3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408,
     3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3426, 3427, 3428, 3429, 3430, 3431,
     3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3452,
     3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3479,
     3487, 3494, 3496, 3499, 3502, 3503, 3504, 3506, 3509, 3511, 3513, 3531, 3533, 3537, 3541, 3543, 3544, 3546, 3583,
     3584, 3587, 3588, 3593, 3594, 3595, 3596, 3598, 3599, 3600, 3601, 3604, 3605, 3611, 3614, 3623, 3624, 3625, 3626,
     3628, 3629, 3630, 3634, 3635, 3636, 3637, 3643, 3644, 3646, 3654, 3655, 3656, 3658, 3659, 3660, 3662, 3663, 3664,
     3665, 3666, 3667, 3670, 3671, 3672, 3673, 3676, 3677, 3678, 3679, 3680, 3681, 3685, 3695, 3697, 3698, 3701, 3703,
     3707, 3709, 3713, 3714, 3715, 3716, 3717, 3722, 3724, 3725, 3726, 3727, 3728, 3730, 3734, 3740, 3761, 3790, 3791,
     3792, 3793, 3795, 3796, 3797, 3798, 3799, 3800, 3805, 3806, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922,
     3923, 3927, 3928], dtype=np.int32)

if __name__ == '__main__':
    print(jaw_mask.shape)
