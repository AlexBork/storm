#include "src/adapters/MathsatExpressionAdapter.h"

bool operator==(msat_decl decl1, msat_decl decl2) {
    return decl1.repr == decl2.repr;
}
