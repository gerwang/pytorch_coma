import numpy as np

front_face_mask = np.array(
    [16, 17, 18, 27, 335, 336, 337, 338, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 477,
     478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 498, 499,
     500, 501, 569, 570, 571, 572, 573, 574, 589, 590, 591, 592, 605, 622, 623, 624, 625, 626, 627, 628,
     629, 630, 667, 668, 669, 670, 671, 672, 673, 674, 679, 680, 681, 682, 683, 688, 691, 692, 693, 694,
     695, 696, 697, 702, 703, 704, 705, 706, 707, 708, 713, 714, 715, 723, 724, 725, 738, 739, 754, 755,
     756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775,
     797, 807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 837, 838, 840, 841,
     842, 846, 847, 848, 864, 865, 877, 878, 879, 880, 881, 882, 883, 884, 885, 896, 897, 898, 899, 902,
     903, 904, 905, 906, 907, 908, 909, 918, 919, 922, 923, 924, 926, 927, 928, 929, 939, 942, 943, 944,
     945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 958, 959, 960, 961, 962, 963, 964, 965, 966,
     967, 968, 969, 970, 971, 972, 977, 978, 979, 980, 985, 986, 991, 992, 993, 994, 995, 999, 1000, 1001,
     1002, 1003, 1006, 1007, 1008, 1010, 1011, 1012, 1013, 1014, 1015, 1019, 1020, 1021, 1022, 1023, 1033,
     1034, 1043, 1044, 1045, 1046, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1068, 1075, 1085, 1086, 1087,
     1088, 1092, 1093, 1096, 1101, 1108, 1113, 1114, 1115, 1116, 1117, 1125, 1126, 1127, 1128, 1129, 1132,
     1134, 1135, 1142, 1143, 1144, 1146, 1147, 1150, 1151, 1152, 1153, 1154, 1155, 1161, 1162, 1163, 1164,
     1168, 1169, 1170, 1175, 1176, 1181, 1182, 1183, 1184, 1189, 1190, 1193, 1194, 1195, 1200, 1201, 1202,
     1216, 1217, 1218, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1232, 1233, 1241, 1242, 1243, 1244, 1283,
     1284, 1287, 1289, 1292, 1293, 1294, 1320, 1321, 1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342,
     1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1361,
     1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378,
     1379, 1380, 1381, 1382, 1383, 1384, 1385, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396,
     1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413,
     1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432,
     1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449,
     1450, 1451, 1452, 1453, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1471,
     1472, 1473, 1474, 1475, 1476, 1477, 1478, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579,
     1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596,
     1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613,
     1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630,
     1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647,
     1648, 1649, 1650, 1651, 1652, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668,
     1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685,
     1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702,
     1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719,
     1720, 1721, 1722, 1723, 1724, 1725, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738,
     1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755,
     1756, 1757, 1758, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1773, 1774, 1775, 1776, 1777,
     1778, 1779, 1780, 1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799,
     1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816,
     1817, 1818, 1819, 1820, 1821, 1823, 1824, 1826, 1827, 1830, 1831, 1832, 1835, 1836, 1846, 1847, 1848,
     1849, 1850, 1851, 1852, 1854, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1869, 1871, 1895, 1982, 1983,
     1984, 1985, 2000, 2001, 2009, 2010, 2011, 2012, 2017, 2018, 2019, 2020, 2021, 2022, 2024, 2030, 2034,
     2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059,
     2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076,
     2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093,
     2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110,
     2111, 2112, 2113, 2114, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2134,
     2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2160, 2161, 2163, 2164, 2165, 2166, 2167, 2168,
     2169, 2170, 2171, 2172, 2173, 2174, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186,
     2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2202, 2203, 2204, 2205,
     2206, 2207, 2220, 2221, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240,
     2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2256, 2264, 2265, 2266, 2267, 2268, 2269, 2270,
     2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287,
     2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304,
     2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321,
     2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338,
     2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355,
     2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372,
     2373, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392,
     2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409,
     2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426,
     2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443,
     2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460,
     2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2478, 2479, 2485, 2486,
     2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503,
     2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520,
     2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2532, 2533, 2534, 2535, 2536, 2537, 2538,
     2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555,
     2556, 2557, 2558, 2559, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574,
     2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2595,
     2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2608, 2609, 2610, 2611, 2612, 2613,
     2614, 2615, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720,
     2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737,
     2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754,
     2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2774, 2775,
     2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792,
     2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809,
     2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826,
     2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843,
     2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860,
     2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878,
     2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2894, 2895, 2896,
     2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913,
     2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2928, 2929, 2930, 2931,
     2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948,
     2949, 2952, 2953, 2973, 3057, 3058, 3059, 3060, 3061, 3062, 3064, 3068, 3070, 3078, 3079, 3080, 3081,
     3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098,
     3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115,
     3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132,
     3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150,
     3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3172, 3173, 3175, 3176, 3177,
     3178, 3179, 3180, 3181, 3182, 3380, 3381, 3384, 3386, 3388, 3390, 3391, 3393, 3394, 3395, 3396, 3397,
     3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414,
     3415, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3464,
     3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481,
     3482, 3483, 3484, 3485, 3487, 3489, 3490, 3491, 3492, 3493, 3495, 3497, 3499, 3500, 3501, 3502, 3503,
     3504, 3505, 3506, 3507, 3508, 3509, 3511, 3512, 3513, 3514, 3515, 3516, 3518, 3520, 3521, 3524, 3526,
     3527, 3531, 3533, 3534, 3537, 3538, 3540, 3541, 3542, 3543, 3546, 3547, 3548, 3549, 3550, 3551, 3552,
     3553, 3555, 3556, 3560, 3561, 3563, 3564, 3571, 3572, 3573, 3575, 3576, 3577, 3580, 3582, 3584, 3585,
     3586, 3588, 3589, 3590, 3591, 3592, 3593, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604,
     3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621,
     3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3638, 3639, 3640, 3641,
     3642, 3645, 3647, 3648, 3651, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665,
     3666, 3667, 3668, 3669, 3670, 3671, 3672, 3674, 3675, 3682, 3683, 3684, 3686, 3687, 3688, 3689, 3690,
     3692, 3694, 3696, 3699, 3700, 3702, 3704, 3705, 3706, 3708, 3710, 3711, 3712, 3714, 3715, 3716, 3717,
     3718, 3720, 3721, 3722, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3737,
     3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3747, 3748, 3749, 3750, 3752, 3753, 3754, 3756, 3759,
     3761, 3762, 3763, 3764, 3766, 3767, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3779, 3780,
     3781, 3782, 3784, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799,
     3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816,
     3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833,
     3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850,
     3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867,
     3868, 3869, 3871, 3872, 3874, 3875, 3876, 3877, 3878, 3880, 3881, 3882, 3884, 3885, 3886, 3887, 3891,
     3892, 3893, 3895, 3896, 3898, 3899, 3900, 3901, 3902, 3903, 3905, 3906, 3907, 3908, 3910, 3911, 3912,
     3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929,
     3930], dtype=np.int32)

if __name__ == '__main__':
    print(front_face_mask.shape)