# Theoretical vs Physical Retrieval Gap Diagnostic Report

**Total Validation True Matches:** 173,390
**Theoretical Config E Attribute Overlap:** 167,370 (96.53%)
**Authoritative Physical Retrieval Hits:** 167,150 (96.40%)
**Gap Delta Pairs:** 811 (0.47%)

---

## Gap Breakdown by Intended Channel

| Intended Channel | Missing Pair Count | % of Ground Truth | Core Mechanism |
| :--- | :---: | :---: | :--- |
| `ch6_distinctive_token` | 288 | 0.17% | Detailed in table below |
| `ch6_or_ch7_bucket_cap` | 257 | 0.15% | Detailed in table below |
| `ch5_char3_k50` | 256 | 0.15% | Detailed in table below |
| `ch7_relaxed_street_num` | 10 | 0.01% | Detailed in table below |

---

## Gap Breakdown by Detailed Failure Reason

| Diagnostic Failure Reason | Missing Pair Count | % of Delta Pairs | Root Cause |
| :--- | :---: | :---: | :--- |
| `token_in_stopword_or_digit` | 258 | 31.81% | Bound by retrieval constraints (top-k, DF cap, bucket limit) |
| `bucket_cap_500_truncated` | 257 | 31.69% | Bound by retrieval constraints (top-k, DF cap, bucket limit) |
| `char_3gram_outside_top50_or_df_cap` | 256 | 31.57% | Bound by retrieval constraints (top-k, DF cap, bucket limit) |
| `shared_token_beyond_top2_rarest` | 30 | 3.70% | Bound by retrieval constraints (top-k, DF cap, bucket limit) |
| `relaxed_street_token_in_stopword_or_cap` | 10 | 1.23% | Bound by retrieval constraints (top-k, DF cap, bucket limit) |

---

## Representative Sample of Delta Pairs

| S1 ID | Target ID | S1 Business Name | Target Business Name | Channel | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `S1-237177847` | `S3-538637647` | Laborers Union Local No 454 | laborersunionlocal.com | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
| `S1-700789055` | `S2-354559016` | Womens Health Federal Center | W0mens Health | `ch6_distinctive_token` | shared_token_beyond_top2_rarest |
| `S1-141265210` | `S3-586411589` | Physical Therapy Group | physicaltherapy.com | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
| `S1-173089268` | `S2-201754355` | He Properties Pvt. Ltd. | He Propertes Pvt. Ltd. | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-173089268` | `S2-203764257` | He Properties Pvt. Ltd. | He Probpeiges Pvt. Ltd. | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-459094286` | `S3-781846004` | Snc Advanced Private Limited | Private Snc Snc Limited-Center | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
| `S1-382769932` | `S2-88326203` | SH Worldwide | SH Center | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-780992647` | `S3-486243235` | Orthopedic Care LLC | 0rthopedic Care L.L.C. | `ch6_or_ch7_bucket_cap` | bucket_cap_500_truncated |
| `S1-21310590` | `S2-544229093` | MG Electronic Corporation | MG CORPORATION CENTER | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-110818108` | `S2-439569782` | S/A Quality | S/A CENTER | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-692731517` | `S3-138548530` | Vision Center Inc | Vision Inc Center Trading | `ch6_or_ch7_bucket_cap` | bucket_cap_500_truncated |
| `S1-692731517` | `S3-803498709` | Vision Center Inc | Vision Centre Inc #52811 | `ch6_or_ch7_bucket_cap` | bucket_cap_500_truncated |
| `S1-570031356` | `S2-105767630` | Heart Center Clinic | heartcenterclinic.com | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
| `S1-273489005` | `S3-99319398` | Golden Constructions Private L | Golden Private Limited Center | `ch6_or_ch7_bucket_cap` | bucket_cap_500_truncated |
| `S1-639086381` | `S3-298467861` | QY Finance Pvt Ltd | QY Pvt Ltd Service | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-59794612` | `S3-87745888` | C+ Constellation LLC | C+ LLC Services | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-417560465` | `S2-239653402` | Phlox & Co | PXH & (CO) | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
| `S1-769943289` | `S2-792823068` | FT Trading Pvt. Ltd. | FT PVT. LTD. CENTER | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-930673187` | `S3-958621131` | Seven International Limited | Limited 5even Inmeratiofnal | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
| `S1-289472075` | `S3-617555756` | Hb & Co | Hb Co Service | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-916500573` | `S2-247049700` | My Exports Private Limited | My Expgots Private Limited | `ch6_distinctive_token` | token_in_stopword_or_digit |
| `S1-662209931` | `S3-155009389` | Office of Veterans Affairs | veteransaffairs.com | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
| `S1-599811277` | `S3-264245556` | Swastik Clinic | Shri swastikclinic.com | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
| `S1-810501556` | `S2-507556632` | Global Silverbox | Global Center | `ch6_or_ch7_bucket_cap` | bucket_cap_500_truncated |
| `S1-359147896` | `S3-89184378` | Classic Producer Private Limit | C1assic Phrecr Private Limited | `ch5_char3_k50` | char_3gram_outside_top50_or_df_cap |
