#ifndef STORM_MODEL_CHECKER_DISCOUNTINGHELPER_H
#define STORM_MODEL_CHECKER_DISCOUNTINGHELPER_H

#include "SingleValueModelCheckerHelper.h"
#include "solver/helper/ValueIterationOperator.h"
#include "storm/storage/Scheduler.h"
#include "utility/ProgressMeasurement.h"

namespace storm {
class Environment;
namespace modelchecker {
namespace helper {
template<typename ValueType>
class DiscountingHelper : public SingleValueModelCheckerHelper<ValueType, storm::models::ModelRepresentation::Sparse> {
   public:
    DiscountingHelper(storm::storage::SparseMatrix<ValueType> const& A);
    DiscountingHelper(storm::storage::SparseMatrix<ValueType> const& A, bool trackScheduler);

    void setUpViOperator() const;

    bool solveWithDiscountedValueIteration(Environment const& env, std::optional<OptimizationDirection> dir, std::vector<ValueType>& x,
                                           std::vector<ValueType> const& b, ValueType discountFactor) const;

    /*!
     * Retrieves the generated scheduler. Note: it is only legal to call this function if a scheduler was generated.
     */
    storm::storage::Scheduler<ValueType> computeScheduler() const;

    /*!
     * Retrieves whether the solver generated a scheduler.
     */
    bool hasScheduler() const;

    void setTrackScheduler(bool trackScheduler);

    bool isTrackSchedulerSet() const;

   private:
    void showProgressIterative(uint64_t iteration) const;

    void extractScheduler(std::vector<ValueType>& x, std::vector<ValueType> const& b, OptimizationDirection const& dir, bool robust) const;

    mutable std::shared_ptr<storm::solver::helper::ValueIterationOperator<ValueType, false>> viOperator;
    mutable std::unique_ptr<std::vector<ValueType>> auxiliaryRowGroupVector;

    mutable boost::optional<storm::utility::ProgressMeasurement> progressMeasurement;

    // If the solver takes posession of the matrix, we store the moved matrix in this member, so it gets deleted
    // when the solver is destructed.
    std::unique_ptr<storm::storage::SparseMatrix<ValueType>> localA;

    // A reference to the original sparse matrix given to this solver. If the solver takes posession of the matrix
    // the reference refers to localA.
    storm::storage::SparseMatrix<ValueType> const* A;

    /// Whether we generate a scheduler during solving.
    bool trackScheduler = false;

    /// The scheduler choices that induce the optimal values (if they could be successfully generated).
    mutable boost::optional<std::vector<uint_fast64_t>> schedulerChoices;
};
}  // namespace helper
}  // namespace modelchecker
}  // namespace storm
#endif  // STORM_MODEL_CHECKER_DISCOUNTINGHELPER_H
