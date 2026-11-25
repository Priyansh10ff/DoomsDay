## Door funnel, n1C S1, maps whose every door is on the path: S1-train-101001, S1-train-101006, S1-train-101009, S1-train-101010, S1-train-101011, S1-train-101014, S1-train-101018, S1-train-101019, S1-train-101020

### all updates (n=231)

| stage of funnel | episodes | share | Wilson 95% |
|---|---|---|---|
| passed >= 1 door(s) | 48 | 20.8% | 0.160-0.265 |
| passed >= 2 door(s) | 4 | 1.7% | 0.007-0.044 |
| passed >= 3 door(s) | 0 | 0.0% | 0.000-0.016 |
| passed all path doors | 24 | 10.4% | |
| exit | 11 | 4.8% | 0.027-0.083 |

After all path doors: 11/24 (45.8%) exit; timeouts 13; failed cells p25/50/75 20/24/32
Success with fewer doors than path doors (should be 0): 0
- path doors 1: n 105, >=1 door 21, exit 10
- path doors 2: n 102, >=1 door 23, exit 1
- path doors 3: n 24, >=1 door 4, exit 0

### u<150 (n=110)

| stage of funnel | episodes | share | Wilson 95% |
|---|---|---|---|
| passed >= 1 door(s) | 33 | 30.0% | 0.222-0.391 |
| passed >= 2 door(s) | 3 | 2.7% | 0.009-0.077 |
| passed >= 3 door(s) | 0 | 0.0% | 0.000-0.034 |
| passed all path doors | 19 | 17.3% | |
| exit | 9 | 8.2% | 0.044-0.148 |

After all path doors: 9/19 (47.4%) exit; timeouts 10; failed cells p25/50/75 20/26/32
Success with fewer doors than path doors (should be 0): 0
- path doors 1: n 49, >=1 door 17, exit 8
- path doors 2: n 49, >=1 door 13, exit 1
- path doors 3: n 12, >=1 door 3, exit 0

### u>=150 (n=121)

| stage of funnel | episodes | share | Wilson 95% |
|---|---|---|---|
| passed >= 1 door(s) | 15 | 12.4% | 0.077-0.194 |
| passed >= 2 door(s) | 1 | 0.8% | 0.001-0.045 |
| passed >= 3 door(s) | 0 | 0.0% | 0.000-0.031 |
| passed all path doors | 5 | 4.1% | |
| exit | 2 | 1.7% | 0.005-0.058 |

After all path doors: 2/5 (40.0%) exit; timeouts 3; failed cells p25/50/75 15/20/33
Success with fewer doors than path doors (should be 0): 0
- path doors 1: n 56, >=1 door 4, exit 2
- path doors 2: n 53, >=1 door 10, exit 0
- path doors 3: n 12, >=1 door 1, exit 0

S1-train-101016 (1 door, off the path): 20/27 (74.1%) exit, dp>=1 in 5
S1 maps with side doors (n=262): dp>=1 79/262 (30.2%), exit 10/262 (3.8%), failed with dp>=path doors (could be side doors) 23

## Matched switch vs pad on S0 (3 rooms, 2 hops)
- gen1 switch ('S0-train-100002', 'S0-train-100004', 'S0-train-100020'): 99/301 (32.9%) (Wilson 0.278-0.384); failed cells p25/50/75 16/20/24; success len p50 532
- gen1 pad ('S0-train-100006', 'S0-train-100007', 'S0-train-100009'): 197/303 (65.0%) (Wilson 0.595-0.702); failed cells p25/50/75 13/19/21; success len p50 403
  gen1 all S0 pad failures: n 628, cells p25/50/75 19/23/32
  gen1 all S0 switch failures: n 723, cells p25/50/75 18/25/37
- n1A switch ('S0-train-100002', 'S0-train-100004', 'S0-train-100020'): 30/431 (7.0%) (Wilson 0.049-0.098); failed cells p25/50/75 21/23/24; success len p50 633
- n1A pad ('S0-train-100006', 'S0-train-100007', 'S0-train-100009'): 411/432 (95.1%) (Wilson 0.927-0.968); failed cells p25/50/75 18/19/24; success len p50 211
  n1A all S0 pad failures: n 421, cells p25/50/75 23/31/41
  n1A all S0 switch failures: n 1204, cells p25/50/75 24/36/48
- n1C switch ('S0-train-100002', 'S0-train-100004', 'S0-train-100020'): 2/22 (9.1%) (Wilson 0.025-0.278); failed cells p25/50/75 19/22/24; success len p50 773
- n1C pad ('S0-train-100006', 'S0-train-100007', 'S0-train-100009'): 20/22 (90.9%) (Wilson 0.722-0.975); failed cells p25/50/75 22/23/23; success len p50 223
  n1C all S0 pad failures: n 28, cells p25/50/75 22/31/40
  n1C all S0 switch failures: n 60, cells p25/50/75 22/31/39
