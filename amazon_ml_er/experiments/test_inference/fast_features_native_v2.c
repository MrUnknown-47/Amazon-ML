#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>

static PyObject *str_alphanumeric_name = NULL;
static PyObject *str_sig_name_tokens = NULL;
static PyObject *str_all_name_tokens = NULL;
static PyObject *str_char2_name = NULL;
static PyObject *str_char3_name = NULL;
static PyObject *str_char4_name = NULL;
static PyObject *str_addr_tokens = NULL;
static PyObject *str_addr_numeric_tokens = NULL;
static PyObject *str_addr_street_num = NULL;
static PyObject *str_addr_postal = NULL;
static PyObject *str_addr_char3 = NULL;
static PyObject *str_is_non_latin = NULL;
static PyObject *str_has_address = NULL;
static PyObject *str_country = NULL;
static PyObject *str_eid = NULL;
static PyObject *str_channels = NULL;
static PyObject *str_rarest_token_df = NULL;

static PyObject *ch1_name = NULL;
static PyObject *ch2_name = NULL;
static PyObject *ch3_name = NULL;
static PyObject *ch4_name = NULL;
static PyObject *ch5_name = NULL;
static PyObject *ch6_name = NULL;
static PyObject *ch7_name = NULL;

static void init_interned_strings(void) {
    if (str_alphanumeric_name) return;
    str_alphanumeric_name = PyUnicode_InternFromString("alphanumeric_name");
    str_sig_name_tokens = PyUnicode_InternFromString("sig_name_tokens");
    str_all_name_tokens = PyUnicode_InternFromString("all_name_tokens");
    str_char2_name = PyUnicode_InternFromString("char2_name");
    str_char3_name = PyUnicode_InternFromString("char3_name");
    str_char4_name = PyUnicode_InternFromString("char4_name");
    str_addr_tokens = PyUnicode_InternFromString("addr_tokens");
    str_addr_numeric_tokens = PyUnicode_InternFromString("addr_numeric_tokens");
    str_addr_street_num = PyUnicode_InternFromString("addr_street_num");
    str_addr_postal = PyUnicode_InternFromString("addr_postal");
    str_addr_char3 = PyUnicode_InternFromString("addr_char3");
    str_is_non_latin = PyUnicode_InternFromString("is_non_latin");
    str_has_address = PyUnicode_InternFromString("has_address");
    str_country = PyUnicode_InternFromString("country");
    str_eid = PyUnicode_InternFromString("eid");
    str_channels = PyUnicode_InternFromString("channels");
    str_rarest_token_df = PyUnicode_InternFromString("rarest_token_df");

    ch1_name = PyUnicode_InternFromString("ch1_core_name");
    ch2_name = PyUnicode_InternFromString("ch2_top2_tokens");
    ch3_name = PyUnicode_InternFromString("ch3_street_num_token");
    ch4_name = PyUnicode_InternFromString("ch4_address_location");
    ch5_name = PyUnicode_InternFromString("ch5_char3_k50");
    ch6_name = PyUnicode_InternFromString("ch6_distinctive_token");
    ch7_name = PyUnicode_InternFromString("ch7_relaxed_street_num");
}

/* Fast Levenshtein distance on two unicode strings */
static PyObject* py_fast_levenshtein_ratio(PyObject* self, PyObject* args) {
    PyObject *s1_obj, *s2_obj;
    if (!PyArg_ParseTuple(args, "OO", &s1_obj, &s2_obj)) return NULL;
    if (!PyUnicode_Check(s1_obj) || !PyUnicode_Check(s2_obj)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be strings");
        return NULL;
    }
    if (s1_obj == s2_obj || PyUnicode_Compare(s1_obj, s2_obj) == 0) return PyFloat_FromDouble(1.0);
    Py_ssize_t len1 = PyUnicode_GET_LENGTH(s1_obj);
    Py_ssize_t len2 = PyUnicode_GET_LENGTH(s2_obj);
    if (len1 == 0 || len2 == 0) return PyFloat_FromDouble(0.0);
    Py_ssize_t max_len = (len1 > len2) ? len1 : len2;
    Py_ssize_t diff = (len1 > len2) ? (len1 - len2) : (len2 - len1);
    if ((double)diff > (double)max_len * 0.7) return PyFloat_FromDouble(0.0);

    int kind1 = PyUnicode_KIND(s1_obj);
    void* data1 = PyUnicode_DATA(s1_obj);
    int kind2 = PyUnicode_KIND(s2_obj);
    void* data2 = PyUnicode_DATA(s2_obj);

    if (len1 > len2) {
        PyObject* ts = s1_obj; s1_obj = s2_obj; s2_obj = ts;
        Py_ssize_t tl = len1; len1 = len2; len2 = tl;
        int tk = kind1; kind1 = kind2; kind2 = tk;
        void* td = data1; data1 = data2; data2 = td;
    }

    int stack1[256], stack2[256];
    int* curr = stack1;
    int* prev = stack2;
    int need_free = 0;
    if (len1 + 1 > 256) {
        curr = (int*)malloc((len1 + 1) * sizeof(int));
        prev = (int*)malloc((len1 + 1) * sizeof(int));
        need_free = 1;
    }
    for (int i = 0; i <= len1; i++) curr[i] = i;

    for (Py_ssize_t i = 0; i < len2; i++) {
        int* temp = prev; prev = curr; curr = temp;
        curr[0] = (int)(i + 1);
        Py_UCS4 c2 = PyUnicode_READ(kind2, data2, i);
        for (Py_ssize_t j = 0; j < len1; j++) {
            Py_UCS4 c1 = PyUnicode_READ(kind1, data1, j);
            int insert = prev[j + 1] + 1;
            int delete = curr[j] + 1;
            int substitute = prev[j] + ((c1 == c2) ? 0 : 1);
            int min_val = (insert < delete) ? insert : delete;
            if (substitute < min_val) min_val = substitute;
            curr[j + 1] = min_val;
        }
    }
    int dist = curr[len1];
    if (need_free) { free(curr); free(prev); }
    double sim = 1.0 - ((double)dist / (double)max_len);
    return PyFloat_FromDouble((sim < 0.0) ? 0.0 : sim);
}

/* Fast intersection count between a tuple of items and a Python set */
static inline Py_ssize_t tuple_set_intersect_count(PyObject* tuple_items, PyObject* target_set) {
    Py_ssize_t n = PyTuple_GET_SIZE(tuple_items);
    Py_ssize_t count = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyTuple_GET_ITEM(tuple_items, i);
        if (PySet_Contains(target_set, item) == 1) {
            count++;
        }
    }
    return count;
}

/* Batch extraction of ranker features for all candidates of s1 into a 2D float32 numpy buffer */
static PyObject* extract_ranker_features_batch_c(PyObject* self, PyObject* args) {
    PyObject *s1_profile;
    PyObject *target_profiles_list;
    PyObject *prov_list;
    PyArrayObject *out_array;
    Py_ssize_t start_row;

    if (!PyArg_ParseTuple(args, "OO!O!O!n",
                          &s1_profile,
                          &PyList_Type, &target_profiles_list,
                          &PyList_Type, &prov_list,
                          &PyArray_Type, &out_array,
                          &start_row)) {
        return NULL;
    }

    Py_ssize_t n_c = PyList_GET_SIZE(target_profiles_list);
    if (n_c == 0) Py_RETURN_NONE;

    init_interned_strings();

    float* out_data = (float*)PyArray_DATA(out_array);
    npy_intp stride_row = PyArray_STRIDE(out_array, 0) / sizeof(float);

    /* Extract S1 attributes ONCE */
    PyObject* s1_alpha_obj = PyObject_GetAttr(s1_profile, str_alphanumeric_name);
    Py_ssize_t s1_alpha_len = 0;
    const char* s1_alpha = PyUnicode_AsUTF8AndSize(s1_alpha_obj, &s1_alpha_len);
    Py_ssize_t s1_char_len = PyUnicode_GET_LENGTH(s1_alpha_obj);

    PyObject* s1_sig_set = PyObject_GetAttr(s1_profile, str_sig_name_tokens);
    PyObject* s1_sig_tuple = PySequence_Tuple(s1_sig_set);
    Py_ssize_t l_s1_sig = PyTuple_GET_SIZE(s1_sig_tuple);

    PyObject* s1_all_toks = PyObject_GetAttr(s1_profile, str_all_name_tokens);
    Py_ssize_t l_s1_all = PySet_Size(s1_all_toks);

    PyObject* s1_c2_set = PyObject_GetAttr(s1_profile, str_char2_name);
    PyObject* s1_c2_tuple = PySequence_Tuple(s1_c2_set);
    Py_ssize_t l_c2_s1 = PyTuple_GET_SIZE(s1_c2_tuple);

    PyObject* s1_c3_set = PyObject_GetAttr(s1_profile, str_char3_name);
    PyObject* s1_c3_tuple = PySequence_Tuple(s1_c3_set);
    Py_ssize_t l_c3_s1 = PyTuple_GET_SIZE(s1_c3_tuple);

    PyObject* s1_c4_set = PyObject_GetAttr(s1_profile, str_char4_name);
    PyObject* s1_c4_tuple = PySequence_Tuple(s1_c4_set);
    Py_ssize_t l_c4_s1 = PyTuple_GET_SIZE(s1_c4_tuple);

    PyObject* s1_addr_toks = PyObject_GetAttr(s1_profile, str_addr_tokens);
    PyObject* s1_addr_tuple = PySequence_Tuple(s1_addr_toks);
    Py_ssize_t l_a1 = PyTuple_GET_SIZE(s1_addr_tuple);

    PyObject* s1_ac3_set = PyObject_GetAttr(s1_profile, str_addr_char3);
    PyObject* s1_ac3_tuple = PySequence_Tuple(s1_ac3_set);
    Py_ssize_t l_ac1 = PyTuple_GET_SIZE(s1_ac3_tuple);

    PyObject* s1_nums = PyObject_GetAttr(s1_profile, str_addr_numeric_tokens);
    PyObject* s1_num0 = (PyList_Check(s1_nums) && PyList_GET_SIZE(s1_nums) > 0) ? PyList_GET_ITEM(s1_nums, 0) : NULL;

    PyObject* s1_street_num = PyObject_GetAttr(s1_profile, str_addr_street_num);
    PyObject* s1_postal = PyObject_GetAttr(s1_profile, str_addr_postal);
    PyObject* s1_country = PyObject_GetAttr(s1_profile, str_country);
    PyObject* s1_non_latin = PyObject_GetAttr(s1_profile, str_is_non_latin);
    int s1_is_non_latin = PyObject_IsTrue(s1_non_latin);
    PyObject* s1_has_addr = PyObject_GetAttr(s1_profile, str_has_address);
    int s1_has_address = PyObject_IsTrue(s1_has_addr);

    /* Loop over targets */
    for (Py_ssize_t i = 0; i < n_c; i++) {
        PyObject* t = PyList_GET_ITEM(target_profiles_list, i);
        PyObject* prov = PyList_GET_ITEM(prov_list, i);
        float* row = out_data + (start_row + i) * stride_row;

        /* Target attributes */
        PyObject* t_alpha_obj = PyObject_GetAttr(t, str_alphanumeric_name);
        Py_ssize_t t_alpha_len = 0;
        const char* t_alpha = PyUnicode_AsUTF8AndSize(t_alpha_obj, &t_alpha_len);
        Py_ssize_t t_char_len = PyUnicode_GET_LENGTH(t_alpha_obj);

        int exact_match = (s1_alpha_len == t_alpha_len && s1_alpha_len > 0 && memcmp(s1_alpha, t_alpha, s1_alpha_len) == 0);
        row[0] = exact_match ? 1.0f : 0.0f;

        /* Significant tokens */
        PyObject* t_sig_set = PyObject_GetAttr(t, str_sig_name_tokens);
        Py_ssize_t l_t_sig = PySet_Size(t_sig_set);
        Py_ssize_t inter_sig = tuple_set_intersect_count(s1_sig_tuple, t_sig_set);
        Py_ssize_t union_sig = l_s1_sig + l_t_sig - inter_sig;
        Py_ssize_t min_sig = (l_s1_sig < l_t_sig) ? l_s1_sig : l_t_sig;
        row[1] = (float)inter_sig / (float)((union_sig > 1) ? union_sig : 1);
        row[2] = (float)inter_sig / (float)((min_sig > 1) ? min_sig : 1);

        /* Char 2-gram */
        PyObject* t_c2_set = PyObject_GetAttr(t, str_char2_name);
        Py_ssize_t l_c2_t = PySet_Size(t_c2_set);
        Py_ssize_t c2_inter = tuple_set_intersect_count(s1_c2_tuple, t_c2_set);
        Py_ssize_t c2_union = l_c2_s1 + l_c2_t - c2_inter;
        row[3] = (float)c2_inter / (float)((c2_union > 1) ? c2_union : 1);

        /* Char 3-gram */
        PyObject* t_c3_set = PyObject_GetAttr(t, str_char3_name);
        Py_ssize_t l_c3_t = PySet_Size(t_c3_set);
        Py_ssize_t c3_inter = tuple_set_intersect_count(s1_c3_tuple, t_c3_set);
        Py_ssize_t c3_union = l_c3_s1 + l_c3_t - c3_inter;
        row[4] = (float)c3_inter / (float)((c3_union > 1) ? c3_union : 1);

        /* Char 4-gram */
        PyObject* t_c4_set = PyObject_GetAttr(t, str_char4_name);
        Py_ssize_t l_c4_t = PySet_Size(t_c4_set);
        Py_ssize_t c4_inter = tuple_set_intersect_count(s1_c4_tuple, t_c4_set);
        Py_ssize_t c4_union = l_c4_s1 + l_c4_t - c4_inter;
        row[5] = (float)c4_inter / (float)((c4_union > 1) ? c4_union : 1);

        /* Length ratio & token count diff */
        Py_ssize_t min_l = (s1_char_len < t_char_len) ? s1_char_len : t_char_len;
        Py_ssize_t max_l = (s1_char_len > t_char_len) ? s1_char_len : t_char_len;
        row[6] = (float)min_l / (float)((max_l > 0) ? max_l : 1);

        PyObject* t_all_toks = PyObject_GetAttr(t, str_all_name_tokens);
        Py_ssize_t l_t_all = PySet_Size(t_all_toks);
        row[7] = (float)labs(l_s1_all - l_t_all);

        /* Address tokens */
        PyObject* t_addr_toks = PyObject_GetAttr(t, str_addr_tokens);
        Py_ssize_t l_a2 = PySet_Size(t_addr_toks);
        Py_ssize_t inter_a = tuple_set_intersect_count(s1_addr_tuple, t_addr_toks);
        Py_ssize_t union_a = l_a1 + l_a2 - inter_a;
        Py_ssize_t min_a = (l_a1 < l_a2) ? l_a1 : l_a2;
        row[8] = (float)inter_a / (float)((union_a > 1) ? union_a : 1);
        row[9] = (float)inter_a / (float)((min_a > 1) ? min_a : 1);

        /* Address char 3-gram */
        PyObject* t_ac3_set = PyObject_GetAttr(t, str_addr_char3);
        Py_ssize_t l_ac2 = PySet_Size(t_ac3_set);
        Py_ssize_t ac3_inter = tuple_set_intersect_count(s1_ac3_tuple, t_ac3_set);
        Py_ssize_t ac3_union = l_ac1 + l_ac2 - ac3_inter;
        row[10] = (float)ac3_inter / (float)((ac3_union > 1) ? ac3_union : 1);

        /* Numeric & street number agreement */
        PyObject* t_nums = PyObject_GetAttr(t, str_addr_numeric_tokens);
        PyObject* t_num0 = (PyList_Check(t_nums) && PyList_GET_SIZE(t_nums) > 0) ? PyList_GET_ITEM(t_nums, 0) : NULL;
        row[11] = (s1_num0 && t_num0 && PyObject_RichCompareBool(s1_num0, t_num0, Py_EQ) == 1) ? 1.0f : 0.0f;

        PyObject* t_street_num = PyObject_GetAttr(t, str_addr_street_num);
        int street_match = (PyUnicode_GET_LENGTH(s1_street_num) > 0 && PyObject_RichCompareBool(s1_street_num, t_street_num, Py_EQ) == 1);
        row[12] = street_match ? 1.0f : 0.0f;

        PyObject* t_postal = PyObject_GetAttr(t, str_addr_postal);
        int postal_match = (PyUnicode_GET_LENGTH(s1_postal) > 0 && PyObject_RichCompareBool(s1_postal, t_postal, Py_EQ) == 1);
        row[13] = postal_match ? 1.0f : 0.0f;

        row[14] = s1_has_address ? 0.0f : 1.0f;
        PyObject* t_has_addr = PyObject_GetAttr(t, str_has_address);
        row[15] = PyObject_IsTrue(t_has_addr) ? 0.0f : 1.0f;

        /* Interaction & consensus */
        row[16] = row[1] * row[8];

        PyObject* t_country = PyObject_GetAttr(t, str_country);
        int same_country = (PyUnicode_GET_LENGTH(s1_country) > 0 && PyObject_RichCompareBool(s1_country, t_country, Py_EQ) == 1);
        row[17] = (float)((exact_match ? 1 : 0) + (street_match ? 1 : 0) + (postal_match ? 1 : 0) + (same_country ? 1 : 0));

        /* Channel Provenance */
        PyObject* channels = (PyDict_Check(prov)) ? PyDict_GetItem(prov, str_channels) : NULL;
        if (channels && PySet_Check(channels)) {
            row[18] = (float)PySet_Size(channels);
            row[19] = (PySet_Contains(channels, ch1_name) == 1) ? 1.0f : 0.0f;
            row[20] = (PySet_Contains(channels, ch2_name) == 1) ? 1.0f : 0.0f;
            row[21] = (PySet_Contains(channels, ch3_name) == 1) ? 1.0f : 0.0f;
            row[22] = (PySet_Contains(channels, ch4_name) == 1) ? 1.0f : 0.0f;
            row[23] = (PySet_Contains(channels, ch5_name) == 1) ? 1.0f : 0.0f;
            row[24] = (PySet_Contains(channels, ch6_name) == 1) ? 1.0f : 0.0f;
            row[25] = (PySet_Contains(channels, ch7_name) == 1) ? 1.0f : 0.0f;
        } else {
            row[18] = row[19] = row[20] = row[21] = row[22] = row[23] = row[24] = row[25] = 0.0f;
        }

        PyObject* rare_obj = (PyDict_Check(prov)) ? PyDict_GetItem(prov, str_rarest_token_df) : NULL;
        long rare_df = (rare_obj && PyLong_Check(rare_obj)) ? PyLong_AsLong(rare_obj) : 0;
        row[26] = (rare_df > 0) ? (float)(1.0 / log(1.0 + (double)rare_df)) : 0.0f;

        PyObject* t_non_latin = PyObject_GetAttr(t, str_is_non_latin);
        int t_is_non_latin = PyObject_IsTrue(t_non_latin);
        row[27] = (s1_is_non_latin != t_is_non_latin) ? 1.0f : 0.0f;

        PyObject* t_eid = PyObject_GetAttr(t, str_eid);
        const char* t_eid_str = PyUnicode_AsUTF8(t_eid);
        row[28] = (t_eid_str && t_eid_str[0] == 'S' && t_eid_str[1] == '3') ? 1.0f : 0.0f;

        /* Cleanup target references */
        Py_DECREF(t_alpha_obj); Py_DECREF(t_sig_set); Py_DECREF(t_all_toks);
        Py_DECREF(t_c2_set); Py_DECREF(t_c3_set); Py_DECREF(t_c4_set);
        Py_DECREF(t_addr_toks); Py_DECREF(t_ac3_set); Py_DECREF(t_nums);
        Py_DECREF(t_street_num); Py_DECREF(t_postal); Py_DECREF(t_has_addr);
        Py_DECREF(t_country); Py_DECREF(t_non_latin); Py_DECREF(t_eid);
    }

    /* Cleanup S1 references */
    Py_DECREF(s1_alpha_obj); Py_DECREF(s1_sig_set); Py_DECREF(s1_sig_tuple);
    Py_DECREF(s1_all_toks); Py_DECREF(s1_c2_set); Py_DECREF(s1_c2_tuple);
    Py_DECREF(s1_c3_set); Py_DECREF(s1_c3_tuple); Py_DECREF(s1_c4_set); Py_DECREF(s1_c4_tuple);
    Py_DECREF(s1_addr_toks); Py_DECREF(s1_addr_tuple); Py_DECREF(s1_ac3_set); Py_DECREF(s1_ac3_tuple);
    Py_DECREF(s1_nums); Py_DECREF(s1_street_num); Py_DECREF(s1_postal);
    Py_DECREF(s1_country); Py_DECREF(s1_non_latin); Py_DECREF(s1_has_addr);

    Py_RETURN_NONE;
}

/* Thread-safe Channel 5 score accumulator using Thread-Local Storage */
static __thread double* tls_ch5_scores = NULL;
static __thread int* tls_ch5_touched = NULL;
static __thread int tls_ch5_capacity = 0;

static PyObject* c_accumulate_scores_dict(PyObject* self, PyObject* args) {
    PyObject* postings_dict;
    PyObject* q_ngrams_seq;
    PyObject* doc_freq_dict;
    PyObject* idf_dict;
    int total_docs;
    int df_threshold;

    if (!PyArg_ParseTuple(args, "OO!O!O!ii",
                          &postings_dict,
                          &PySet_Type, &q_ngrams_seq,
                          &PyDict_Type, &doc_freq_dict,
                          &PyDict_Type, &idf_dict,
                          &total_docs, &df_threshold)) {
        return NULL;
    }

    PyObject* q_tuple = PySequence_Tuple(q_ngrams_seq);
    Py_ssize_t n_q = PyTuple_GET_SIZE(q_tuple);
    if (n_q == 0) { Py_DECREF(q_tuple); return PyDict_New(); }

    if (total_docs > tls_ch5_capacity) {
        if (tls_ch5_scores) free(tls_ch5_scores);
        if (tls_ch5_touched) free(tls_ch5_touched);
        tls_ch5_capacity = total_docs + 10000;
        tls_ch5_scores = (double*)calloc(tls_ch5_capacity, sizeof(double));
        tls_ch5_touched = (int*)malloc(tls_ch5_capacity * sizeof(int));
    }

    int num_touched = 0;

    for (Py_ssize_t i = 0; i < n_q; i++) {
        PyObject* ng = PyTuple_GET_ITEM(q_tuple, i);
        PyObject* df_obj = PyDict_GetItem(doc_freq_dict, ng);
        if (!df_obj) continue;
        long df = PyLong_AsLong(df_obj);
        if (df == 0 || (df_threshold > 0 && df > df_threshold)) continue;

        double w;
        if (df_threshold > 0) {
            w = log(1.0 + ((double)total_docs / (double)df));
        } else {
            PyObject* idf_obj = PyDict_GetItem(idf_dict, ng);
            if (!idf_obj) continue;
            w = PyFloat_AsDouble(idf_obj);
        }

        PyObject* post_list = PyDict_GetItem(postings_dict, ng);
        if (!post_list || !PyList_Check(post_list)) continue;

        Py_ssize_t n_post = PyList_GET_SIZE(post_list);
        for (Py_ssize_t j = 0; j < n_post; j++) {
            long int_id = PyLong_AsLong(PyList_GET_ITEM(post_list, j));
            if (int_id >= 0 && int_id < total_docs) {
                if (tls_ch5_scores[int_id] == 0.0) {
                    tls_ch5_touched[num_touched++] = (int)int_id;
                }
                tls_ch5_scores[int_id] += w;
            }
        }
    }
    Py_DECREF(q_tuple);

    PyObject* dict = PyDict_New();
    for (int i = 0; i < num_touched; i++) {
        int id = tls_ch5_touched[i];
        PyObject* k = PyLong_FromLong(id);
        PyObject* v = PyFloat_FromDouble(tls_ch5_scores[id]);
        PyDict_SetItem(dict, k, v);
        Py_DECREF(k);
        Py_DECREF(v);
        tls_ch5_scores[id] = 0.0; /* Reset for next query */
    }

    return dict;
}

/* ========================================================================= */
/* FAST NATIVE MULTI-THREAD RANKER FEATURE EXTRACTION                        */
/* ========================================================================= */

static inline uint64_t fnv1a_64(const char* str, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)(unsigned char)str[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int compare_uint64(const void* a, const void* b) {
    uint64_t va = *(const uint64_t*)a;
    uint64_t vb = *(const uint64_t*)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

static inline int intersect_sorted_64(const uint64_t* a, int na, const uint64_t* b, int nb) {
    int i = 0, j = 0, count = 0;
    while (i < na && j < nb) {
        uint64_t va = a[i];
        uint64_t vb = b[j];
        if (va == vb) {
            count++;
            i++;
            j++;
        } else if (va < vb) {
            i++;
        } else {
            j++;
        }
    }
    return count;
}

typedef struct {
    uint32_t alpha_len;
    uint32_t char_len;
    uint64_t alpha_hash;
    
    uint16_t n_sig;
    uint16_t n_all;
    uint16_t n_c2;
    uint16_t n_c3;
    uint16_t n_c4;
    uint16_t n_atok;
    uint16_t n_ac3;
    
    uint64_t num0_hash;
    uint64_t street_num_hash;
    uint64_t postal_hash;
    uint64_t country_hash;
    
    uint8_t has_address;
    uint8_t is_non_latin;
    uint8_t is_s3;
    
    uint64_t* sig_tokens;
    uint64_t* c2;
    uint64_t* c3;
    uint64_t* c4;
    uint64_t* atok;
    uint64_t* ac3;
    char* alpha_str;
} NativeProfile;

static uint64_t* pack_set_of_strings(PyObject* set_obj, uint16_t* out_len) {
    if (!set_obj || !PySet_Check(set_obj)) {
        *out_len = 0;
        return NULL;
    }
    Py_ssize_t size = PySet_Size(set_obj);
    if (size == 0) {
        *out_len = 0;
        return NULL;
    }
    uint64_t* arr = (uint64_t*)malloc(size * sizeof(uint64_t));
    if (!arr) {
        *out_len = 0;
        return NULL;
    }
    PyObject* iterator = PyObject_GetIter(set_obj);
    PyObject* item;
    Py_ssize_t idx = 0;
    if (iterator) {
        while ((item = PyIter_Next(iterator))) {
            if (PyUnicode_Check(item)) {
                Py_ssize_t str_len;
                const char* str_utf8 = PyUnicode_AsUTF8AndSize(item, &str_len);
                if (str_utf8) {
                    arr[idx++] = fnv1a_64(str_utf8, (size_t)str_len);
                }
            }
            Py_DECREF(item);
        }
        Py_DECREF(iterator);
    }
    qsort(arr, idx, sizeof(uint64_t), compare_uint64);
    *out_len = (uint16_t)idx;
    return arr;
}

static void pack_native_profile(PyObject* prof, NativeProfile* np) {
    init_interned_strings();
    
    // Alphanumeric name
    PyObject* alpha_obj = PyObject_GetAttr(prof, str_alphanumeric_name);
    if (alpha_obj && PyUnicode_Check(alpha_obj)) {
        Py_ssize_t slen = 0;
        const char* s = PyUnicode_AsUTF8AndSize(alpha_obj, &slen);
        np->alpha_len = (uint32_t)slen;
        np->char_len = (uint32_t)PyUnicode_GET_LENGTH(alpha_obj);
        if (slen > 0 && s) {
            np->alpha_hash = fnv1a_64(s, (size_t)slen);
            np->alpha_str = (char*)malloc(slen + 1);
            memcpy(np->alpha_str, s, slen);
            np->alpha_str[slen] = '\0';
        } else {
            np->alpha_hash = 0;
            np->alpha_str = NULL;
        }
    } else {
        np->alpha_len = 0;
        np->char_len = 0;
        np->alpha_hash = 0;
        np->alpha_str = NULL;
    }
    Py_XDECREF(alpha_obj);

    // all_name_tokens count
    PyObject* all_toks = PyObject_GetAttr(prof, str_all_name_tokens);
    np->n_all = (all_toks && PySet_Check(all_toks)) ? (uint16_t)PySet_Size(all_toks) : 0;
    Py_XDECREF(all_toks);

    // Sets: sig_name_tokens, char2_name, char3_name, char4_name, addr_tokens, addr_char3
    PyObject* sig_set = PyObject_GetAttr(prof, str_sig_name_tokens);
    np->sig_tokens = pack_set_of_strings(sig_set, &np->n_sig);
    Py_XDECREF(sig_set);

    PyObject* c2_set = PyObject_GetAttr(prof, str_char2_name);
    np->c2 = pack_set_of_strings(c2_set, &np->n_c2);
    Py_XDECREF(c2_set);

    PyObject* c3_set = PyObject_GetAttr(prof, str_char3_name);
    np->c3 = pack_set_of_strings(c3_set, &np->n_c3);
    Py_XDECREF(c3_set);

    PyObject* c4_set = PyObject_GetAttr(prof, str_char4_name);
    np->c4 = pack_set_of_strings(c4_set, &np->n_c4);
    Py_XDECREF(c4_set);

    PyObject* atok_set = PyObject_GetAttr(prof, str_addr_tokens);
    np->atok = pack_set_of_strings(atok_set, &np->n_atok);
    Py_XDECREF(atok_set);

    PyObject* ac3_set = PyObject_GetAttr(prof, str_addr_char3);
    np->ac3 = pack_set_of_strings(ac3_set, &np->n_ac3);
    Py_XDECREF(ac3_set);

    // addr_numeric_tokens (first item)
    PyObject* nums = PyObject_GetAttr(prof, str_addr_numeric_tokens);
    np->num0_hash = 0;
    if (nums && PyList_Check(nums) && PyList_GET_SIZE(nums) > 0) {
        PyObject* item0 = PyList_GET_ITEM(nums, 0);
        if (item0 && PyUnicode_Check(item0)) {
            Py_ssize_t slen = 0;
            const char* s = PyUnicode_AsUTF8AndSize(item0, &slen);
            if (slen > 0 && s) np->num0_hash = fnv1a_64(s, (size_t)slen);
        }
    }
    Py_XDECREF(nums);

    // street_num
    PyObject* st = PyObject_GetAttr(prof, str_addr_street_num);
    np->street_num_hash = 0;
    if (st && PyUnicode_Check(st)) {
        Py_ssize_t slen = 0;
        const char* s = PyUnicode_AsUTF8AndSize(st, &slen);
        if (slen > 0 && s) np->street_num_hash = fnv1a_64(s, (size_t)slen);
    }
    Py_XDECREF(st);

    // postal
    PyObject* post = PyObject_GetAttr(prof, str_addr_postal);
    np->postal_hash = 0;
    if (post && PyUnicode_Check(post)) {
        Py_ssize_t slen = 0;
        const char* s = PyUnicode_AsUTF8AndSize(post, &slen);
        if (slen > 0 && s) np->postal_hash = fnv1a_64(s, (size_t)slen);
    }
    Py_XDECREF(post);

    // country
    PyObject* ctry = PyObject_GetAttr(prof, str_country);
    np->country_hash = 0;
    if (ctry && PyUnicode_Check(ctry)) {
        Py_ssize_t slen = 0;
        const char* s = PyUnicode_AsUTF8AndSize(ctry, &slen);
        if (slen > 0 && s) np->country_hash = fnv1a_64(s, (size_t)slen);
    }
    Py_XDECREF(ctry);

    // flags
    PyObject* has_a = PyObject_GetAttr(prof, str_has_address);
    np->has_address = (has_a && PyObject_IsTrue(has_a)) ? 1 : 0;
    Py_XDECREF(has_a);

    PyObject* non_l = PyObject_GetAttr(prof, str_is_non_latin);
    np->is_non_latin = (non_l && PyObject_IsTrue(non_l)) ? 1 : 0;
    Py_XDECREF(non_l);

    PyObject* eid_obj = PyObject_GetAttr(prof, str_eid);
    np->is_s3 = 0;
    if (eid_obj && PyUnicode_Check(eid_obj)) {
        const char* s = PyUnicode_AsUTF8(eid_obj);
        if (s && s[0] == 'S' && s[1] == '3') np->is_s3 = 1;
    }
    Py_XDECREF(eid_obj);
}

static void free_native_profile(NativeProfile* np) {
    if (np->alpha_str) free(np->alpha_str);
    if (np->sig_tokens) free(np->sig_tokens);
    if (np->c2) free(np->c2);
    if (np->c3) free(np->c3);
    if (np->c4) free(np->c4);
    if (np->atok) free(np->atok);
    if (np->ac3) free(np->ac3);
}

static inline void compute_ranker_features(
    const NativeProfile* s1,
    const NativeProfile* t,
    uint8_t channels_mask,
    uint32_t rare_df,
    float* row
) {
    // 0: name_exact_match
    int exact_match = 0;
    if (s1->alpha_len == t->alpha_len && s1->alpha_len > 0) {
        if (s1->alpha_hash == t->alpha_hash) {
            if (s1->alpha_str && t->alpha_str && memcmp(s1->alpha_str, t->alpha_str, s1->alpha_len) == 0) {
                exact_match = 1;
            }
        }
    }
    row[0] = exact_match ? 1.0f : 0.0f;

    // 1: name_token_jaccard, 2: name_token_containment
    int inter_sig = intersect_sorted_64(s1->sig_tokens, s1->n_sig, t->sig_tokens, t->n_sig);
    int union_sig = (int)s1->n_sig + (int)t->n_sig - inter_sig;
    int min_sig = (s1->n_sig < t->n_sig) ? (int)s1->n_sig : (int)t->n_sig;
    row[1] = (float)inter_sig / (float)((union_sig > 1) ? union_sig : 1);
    row[2] = (float)inter_sig / (float)((min_sig > 1) ? min_sig : 1);

    // 3: name_char2_sim
    int c2_inter = intersect_sorted_64(s1->c2, s1->n_c2, t->c2, t->n_c2);
    int c2_union = (int)s1->n_c2 + (int)t->n_c2 - c2_inter;
    row[3] = (float)c2_inter / (float)((c2_union > 1) ? c2_union : 1);

    // 4: name_char3_sim
    int c3_inter = intersect_sorted_64(s1->c3, s1->n_c3, t->c3, t->n_c3);
    int c3_union = (int)s1->n_c3 + (int)t->n_c3 - c3_inter;
    row[4] = (float)c3_inter / (float)((c3_union > 1) ? c3_union : 1);

    // 5: name_char4_sim
    int c4_inter = intersect_sorted_64(s1->c4, s1->n_c4, t->c4, t->n_c4);
    int c4_union = (int)s1->n_c4 + (int)t->n_c4 - c4_inter;
    row[5] = (float)c4_inter / (float)((c4_union > 1) ? c4_union : 1);

    // 6: name_len_ratio
    uint32_t min_l = (s1->char_len < t->char_len) ? s1->char_len : t->char_len;
    uint32_t max_l = (s1->char_len > t->char_len) ? s1->char_len : t->char_len;
    row[6] = (float)min_l / (float)((max_l > 0) ? max_l : 1);

    // 7: name_token_count_diff
    row[7] = (float)abs((int)s1->n_all - (int)t->n_all);

    // 8: addr_token_jaccard, 9: addr_token_containment
    int inter_a = intersect_sorted_64(s1->atok, s1->n_atok, t->atok, t->n_atok);
    int union_a = (int)s1->n_atok + (int)t->n_atok - inter_a;
    int min_a = (s1->n_atok < t->n_atok) ? (int)s1->n_atok : (int)t->n_atok;
    row[8] = (float)inter_a / (float)((union_a > 1) ? union_a : 1);
    row[9] = (float)inter_a / (float)((min_a > 1) ? min_a : 1);

    // 10: addr_char3_sim
    int ac3_inter = intersect_sorted_64(s1->ac3, s1->n_ac3, t->ac3, t->n_ac3);
    int ac3_union = (int)s1->n_ac3 + (int)t->n_ac3 - ac3_inter;
    row[10] = (float)ac3_inter / (float)((ac3_union > 1) ? ac3_union : 1);

    // 11: addr_num_agreement
    row[11] = (s1->num0_hash != 0 && s1->num0_hash == t->num0_hash) ? 1.0f : 0.0f;

    // 12: addr_street_num_match
    int street_match = (s1->street_num_hash != 0 && s1->street_num_hash == t->street_num_hash);
    row[12] = street_match ? 1.0f : 0.0f;

    // 13: addr_postal_match
    int postal_match = (s1->postal_hash != 0 && s1->postal_hash == t->postal_hash);
    row[13] = postal_match ? 1.0f : 0.0f;

    // 14: addr_missing_s1, 15: addr_missing_t
    row[14] = s1->has_address ? 0.0f : 1.0f;
    row[15] = t->has_address ? 0.0f : 1.0f;

    // 16: name_x_addr_sim
    row[16] = row[1] * row[8];

    // 17: exact_agreement_count
    int same_country = (s1->country_hash != 0 && s1->country_hash == t->country_hash);
    row[17] = (float)((exact_match ? 1 : 0) + (street_match ? 1 : 0) + (postal_match ? 1 : 0) + (same_country ? 1 : 0));

    // 18..25: Provenance
    row[18] = (float)__builtin_popcount((unsigned int)channels_mask);
    row[19] = (channels_mask & 1) ? 1.0f : 0.0f;
    row[20] = (channels_mask & 2) ? 1.0f : 0.0f;
    row[21] = (channels_mask & 4) ? 1.0f : 0.0f;
    row[22] = (channels_mask & 8) ? 1.0f : 0.0f;
    row[23] = (channels_mask & 16) ? 1.0f : 0.0f;
    row[24] = (channels_mask & 32) ? 1.0f : 0.0f;
    row[25] = (channels_mask & 64) ? 1.0f : 0.0f;

    // 26: rare_token_score
    row[26] = (rare_df > 0) ? (float)(1.0 / log(1.0 + (double)rare_df)) : 0.0f;

    // 27: is_transliteration_pair
    row[27] = (s1->is_non_latin != t->is_non_latin) ? 1.0f : 0.0f;

    // 28: target_is_source3
    row[28] = t->is_s3 ? 1.0f : 0.0f;
}

typedef struct {
    size_t start_idx;
    size_t end_idx;
    const NativeProfile* s1_profiles;
    const NativeProfile* target_profiles;
    const uint32_t* pair_s1;
    const uint32_t* pair_t;
    const uint8_t* pair_channels;
    const uint32_t* pair_rare;
    float* out_data;
    npy_intp stride_row;
} WorkerArgs;

static void* worker_thread_func(void* arg) {
    WorkerArgs* w = (WorkerArgs*)arg;
    for (size_t i = w->start_idx; i < w->end_idx; i++) {
        uint32_t s1_i = w->pair_s1[i];
        uint32_t t_i = w->pair_t[i];
        const NativeProfile* s1 = &w->s1_profiles[s1_i];
        const NativeProfile* t = &w->target_profiles[t_i];
        float* row = w->out_data + i * w->stride_row;
        compute_ranker_features(s1, t, w->pair_channels[i], w->pair_rare[i], row);
    }
    return NULL;
}

/* Parallel batch extraction of ranker features across native pthreads */
static PyObject* extract_ranker_features_parallel_c(PyObject* self, PyObject* args) {
    PyObject* s1_profiles_list;
    PyObject* target_profiles_list;
    PyArrayObject* pair_s1_arr;
    PyArrayObject* pair_t_arr;
    PyArrayObject* pair_chan_arr;
    PyArrayObject* pair_rare_arr;
    PyArrayObject* out_arr;
    int num_threads;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!i",
                          &PyList_Type, &s1_profiles_list,
                          &PyList_Type, &target_profiles_list,
                          &PyArray_Type, &pair_s1_arr,
                          &PyArray_Type, &pair_t_arr,
                          &PyArray_Type, &pair_chan_arr,
                          &PyArray_Type, &pair_rare_arr,
                          &PyArray_Type, &out_arr,
                          &num_threads)) {
        return NULL;
    }

    Py_ssize_t n_pairs = PyArray_DIM(pair_s1_arr, 0);
    if (n_pairs == 0) Py_RETURN_NONE;

    Py_ssize_t n_s1 = PyList_GET_SIZE(s1_profiles_list);
    Py_ssize_t n_targets = PyList_GET_SIZE(target_profiles_list);

    // Pack S1 profiles
    NativeProfile* s1_native = (NativeProfile*)malloc(n_s1 * sizeof(NativeProfile));
    if (!s1_native) { PyErr_NoMemory(); return NULL; }
    for (Py_ssize_t i = 0; i < n_s1; i++) {
        pack_native_profile(PyList_GET_ITEM(s1_profiles_list, i), &s1_native[i]);
    }

    // Pack Target profiles
    NativeProfile* t_native = (NativeProfile*)malloc(n_targets * sizeof(NativeProfile));
    if (!t_native) {
        for (Py_ssize_t i = 0; i < n_s1; i++) free_native_profile(&s1_native[i]);
        free(s1_native);
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n_targets; i++) {
        pack_native_profile(PyList_GET_ITEM(target_profiles_list, i), &t_native[i]);
    }

    const uint32_t* p_s1 = (const uint32_t*)PyArray_DATA(pair_s1_arr);
    const uint32_t* p_t = (const uint32_t*)PyArray_DATA(pair_t_arr);
    const uint8_t* p_chan = (const uint8_t*)PyArray_DATA(pair_chan_arr);
    const uint32_t* p_rare = (const uint32_t*)PyArray_DATA(pair_rare_arr);
    float* out_data = (float*)PyArray_DATA(out_arr);
    npy_intp stride_row = PyArray_STRIDE(out_arr, 0) / sizeof(float);

    if (num_threads <= 1 || n_pairs < 2000) {
        // Run single-threaded
        Py_BEGIN_ALLOW_THREADS
        for (size_t i = 0; i < (size_t)n_pairs; i++) {
            uint32_t s1_i = p_s1[i];
            uint32_t t_i = p_t[i];
            float* row = out_data + i * stride_row;
            compute_ranker_features(&s1_native[s1_i], &t_native[t_i], p_chan[i], p_rare[i], row);
        }
        Py_END_ALLOW_THREADS
    } else {
        if (num_threads > 16) num_threads = 16;
        pthread_t threads[16];
        WorkerArgs args[16];
        size_t chunk_size = ((size_t)n_pairs + num_threads - 1) / num_threads;

        Py_BEGIN_ALLOW_THREADS
        for (int t = 0; t < num_threads; t++) {
            size_t s = t * chunk_size;
            size_t e = s + chunk_size;
            if (s > (size_t)n_pairs) s = (size_t)n_pairs;
            if (e > (size_t)n_pairs) e = (size_t)n_pairs;
            args[t].start_idx = s;
            args[t].end_idx = e;
            args[t].s1_profiles = s1_native;
            args[t].target_profiles = t_native;
            args[t].pair_s1 = p_s1;
            args[t].pair_t = p_t;
            args[t].pair_channels = p_chan;
            args[t].pair_rare = p_rare;
            args[t].out_data = out_data;
            args[t].stride_row = stride_row;

            if (s < e) {
                pthread_create(&threads[t], NULL, worker_thread_func, &args[t]);
            }
        }

        for (int t = 0; t < num_threads; t++) {
            if (args[t].start_idx < args[t].end_idx) {
                pthread_join(threads[t], NULL);
            }
        }
        Py_END_ALLOW_THREADS
    }

    // Cleanup
    for (Py_ssize_t i = 0; i < n_s1; i++) free_native_profile(&s1_native[i]);
    free(s1_native);
    for (Py_ssize_t i = 0; i < n_targets; i++) free_native_profile(&t_native[i]);
    free(t_native);

    Py_RETURN_NONE;
}

/* Method definitions */
static PyMethodDef FastFeatureMethods[] = {
    {"extract_ranker_features_batch_c", extract_ranker_features_batch_c, METH_VARARGS, "Batch extract ranker features in C"},
    {"extract_ranker_features_parallel_c", extract_ranker_features_parallel_c, METH_VARARGS, "Parallel batch extract ranker features in native C"},
    {"fast_levenshtein_ratio", py_fast_levenshtein_ratio, METH_VARARGS, "Fast unicode Levenshtein ratio in C"},
    {"c_accumulate_scores_dict", c_accumulate_scores_dict, METH_VARARGS, "Fast Channel 5 accumulator in C"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastfeaturesmodule = {
    PyModuleDef_HEAD_INIT, "fast_features_native_v2", NULL, -1, FastFeatureMethods
};

PyMODINIT_FUNC PyInit_fast_features_native_v2(void) {
    import_array();
    init_interned_strings();
    return PyModule_Create(&fastfeaturesmodule);
}
