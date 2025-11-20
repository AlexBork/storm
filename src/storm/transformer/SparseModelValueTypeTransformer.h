#pragma once
#include "storm/models/sparse/Model.h"
#include "storm/storage/sparse/ModelComponents.h"

namespace storm::transformer {

template<typename InputValueType, typename OutputValueType>
class SparseModelValueTypeTransformer {
   public:
    explicit SparseModelValueTypeTransformer() = default;

    std::shared_ptr<storm::models::sparse::Model<OutputValueType>> transformModel(std::shared_ptr<models::sparse::Model<InputValueType>> const& inputModel);
};

}  // namespace storm::transformer
