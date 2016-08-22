void mtbdd_getsha(MTBDD mtbdd, char *target); // target must be at least 65 bytes...

/**
 * Binary operation Divide (for MTBDDs of same type)
 * Only for MTBDDs where all leaves are Integer or Double.
 * If either operand is mtbdd_false (not defined),
 * then the result is mtbdd_false (i.e. not defined).
 */
TASK_DECL_2(MTBDD, mtbdd_op_divide, MTBDD*, MTBDD*)
#define mtbdd_divide(a, b) mtbdd_apply(a, b, TASK(mtbdd_op_divide))

/**
 * Binary operation equals (for MTBDDs of same type)
 * Only for MTBDDs where either all leaves are Boolean, or Integer, or Double.
 * For Integer/Double MTBDD, if either operand is mtbdd_false (not defined),
 * then the result is the other operand.
 */
TASK_DECL_2(MTBDD, mtbdd_op_equals, MTBDD*, MTBDD*)
#define mtbdd_equals(a, b) mtbdd_apply(a, b, TASK(mtbdd_op_equals))

/**
 * Binary operation Less (for MTBDDs of same type)
 * Only for MTBDDs where either all leaves are Boolean, or Integer, or Double.
 * For Integer/Double MTBDD, if either operand is mtbdd_false (not defined),
 * then the result is the other operand.
 */
TASK_DECL_2(MTBDD, mtbdd_op_less, MTBDD*, MTBDD*)
#define mtbdd_less_as_bdd(a, b) mtbdd_apply(a, b, TASK(mtbdd_op_less))

/**
 * Binary operation Less (for MTBDDs of same type)
 * Only for MTBDDs where either all leaves are Boolean, or Integer, or Double.
 * For Integer/Double MTBDD, if either operand is mtbdd_false (not defined),
 * then the result is the other operand.
 */
TASK_DECL_2(MTBDD, mtbdd_op_less_or_equal, MTBDD*, MTBDD*)
#define mtbdd_less_or_equal_as_bdd(a, b) mtbdd_apply(a, b, TASK(mtbdd_op_less_or_equal))

/**
 * Binary operation Pow (for MTBDDs of same type)
 * Only for MTBDDs where either all leaves are Integer, Double or a Fraction.
 * For Integer/Double MTBDD, if either operand is mtbdd_false (not defined),
 * then the result is the other operand.
 */
TASK_DECL_2(MTBDD, mtbdd_op_pow, MTBDD*, MTBDD*)
#define mtbdd_pow(a, b) mtbdd_apply(a, b, TASK(mtbdd_op_pow))

/**
 * Binary operation Mod (for MTBDDs of same type)
 * Only for MTBDDs where either all leaves are Integer, Double or a Fraction.
 * For Integer/Double MTBDD, if either operand is mtbdd_false (not defined),
 * then the result is the other operand.
 */
TASK_DECL_2(MTBDD, mtbdd_op_mod, MTBDD*, MTBDD*)
#define mtbdd_mod(a, b) mtbdd_apply(a, b, TASK(mtbdd_op_mod))

/**
 * Binary operation Log (for MTBDDs of same type)
 * Only for MTBDDs where either all leaves are Double or a Fraction.
 * For Integer/Double MTBDD, if either operand is mtbdd_false (not defined),
 * then the result is the other operand.
 */
TASK_DECL_2(MTBDD, mtbdd_op_logxy, MTBDD*, MTBDD*)
#define mtbdd_logxy(a, b) mtbdd_apply(a, b, TASK(mtbdd_op_logxy))

/**
 * Monad that converts double to a Boolean MTBDD, translate terminals != 0 to 1 and to 0 otherwise;
 */
TASK_DECL_2(MTBDD, mtbdd_op_not_zero, MTBDD, size_t)
TASK_DECL_1(MTBDD, mtbdd_not_zero, MTBDD)
#define mtbdd_not_zero(dd) CALL(mtbdd_not_zero, dd)

/**
 * Monad that floors all Double and Fraction values.
 */
TASK_DECL_2(MTBDD, mtbdd_op_floor, MTBDD, size_t)
TASK_DECL_1(MTBDD, mtbdd_floor, MTBDD)
#define mtbdd_floor(dd) CALL(mtbdd_floor, dd)

/**
 * Monad that ceils all Double and Fraction values.
 */
TASK_DECL_2(MTBDD, mtbdd_op_ceil, MTBDD, size_t)
TASK_DECL_1(MTBDD, mtbdd_ceil, MTBDD)
#define mtbdd_ceil(dd) CALL(mtbdd_ceil, dd)

/**
 * Monad that converts Boolean to a Double MTBDD, translate terminals true to 1 and to 0 otherwise;
 */
TASK_DECL_2(MTBDD, mtbdd_op_bool_to_double, MTBDD, size_t)
TASK_DECL_1(MTBDD, mtbdd_bool_to_double, MTBDD)
#define mtbdd_bool_to_double(dd) CALL(mtbdd_bool_to_double, dd)

/**
 * Monad that converts Boolean to a uint MTBDD, translate terminals true to 1 and to 0 otherwise;
 */
TASK_DECL_2(MTBDD, mtbdd_op_bool_to_int64, MTBDD, size_t)
TASK_DECL_1(MTBDD, mtbdd_bool_to_int64, MTBDD)
#define mtbdd_bool_to_int64(dd) CALL(mtbdd_bool_to_int64, dd)

/**
 * Count the number of assignments (minterms) leading to a non-zero
 */
TASK_DECL_2(double, mtbdd_non_zero_count, MTBDD, size_t)
#define mtbdd_non_zero_count(dd, nvars) CALL(mtbdd_non_zero_count, dd, nvars)

// Checks whether the given MTBDD (does represents a zero leaf.
int mtbdd_iszero(MTBDD);
int mtbdd_isnonzero(MTBDD);

#define mtbdd_regular(dd) (dd & ~mtbdd_complement)

#define mtbdd_set_next(set) (mtbdd_gethigh(set))
#define mtbdd_set_isempty(set) (set == mtbdd_true)

/* Create a MTBDD representing just <var> or the negation of <var> */
MTBDD mtbdd_ithvar(uint32_t var);

/**
 * Unary operation Complement.
 * Supported domains: Integer, Real
 */
TASK_DECL_2(MTBDD, mtbdd_op_complement, MTBDD, size_t);

/**
 * Compute the complement of a.
 */
#define mtbdd_get_complement(a) mtbdd_uapply(a, TASK(mtbdd_op_complement), 0)

/**
 * Just like mtbdd_abstract_min, but instead of abstracting the variables in the given cube, picks a unique representative that realizes the minimal function value.
 */
TASK_DECL_3(MTBDD, mtbdd_minExistsRepresentative, MTBDD, MTBDD, uint32_t);
#define mtbdd_minExistsRepresentative(a, vars) (CALL(mtbdd_minExistsRepresentative, a, vars, 0))

/**
 * Just like mtbdd_abstract_max but instead of abstracting the variables in the given cube, picks a unique representative that realizes the maximal function value.
 */
TASK_DECL_3(MTBDD, mtbdd_maxExistsRepresentative, MTBDD, MTBDD, uint32_t);
#define mtbdd_maxExistsRepresentative(a, vars) (CALL(mtbdd_maxExistsRepresentative, a, vars, 0))

