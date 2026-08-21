#pragma once

#include <memory>

#include "storm/adapters/RationalNumberForward.h"
#include "storm/models/sparse/Model.h"

namespace storm::transformer {

/**
 * Returns a model equivalent to @p inputModel with all probabilities, rates, and rewards converted to double.
 *
 * @pre inputModel is not null.
 */
static std::shared_ptr<storm::models::sparse::Model<double>> sparseRationalModelToDouble(
    std::shared_ptr<models::sparse::Model<storm::RationalNumber>> const& inputModel);

}  // namespace storm::transformer
