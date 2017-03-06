#pragma once

#include <vector>
#include <memory>
#include <boost/optional.hpp>

#include "storm/modelchecker/parametric/SparseParameterLiftingModelChecker.h"
#include "storm/storage/BitVector.h"
#include "storm/solver/MinMaxLinearEquationSolver.h"
#include "storm/transformer/ParameterLifter.h"

namespace storm {
    namespace modelchecker {
        namespace parametric {
            
            template <typename SparseModelType, typename ConstantType>
            class SparseDtmcParameterLiftingModelChecker : public SparseParameterLiftingModelChecker<SparseModelType, ConstantType> {
            public:
                SparseDtmcParameterLiftingModelChecker(SparseModelType const& parametricModel);
                SparseDtmcParameterLiftingModelChecker(SparseModelType const& parametricModel, std::unique_ptr<storm::solver::MinMaxLinearEquationSolverFactory<ConstantType>>&& solverFactory);
                
            protected:
                
                virtual void specifyBoundedUntilFormula(CheckTask<storm::logic::BoundedUntilFormula, ConstantType> const& checkTask) override;
                virtual void specifyUntilFormula(CheckTask<storm::logic::UntilFormula, ConstantType> const& checkTask) override;
                virtual void specifyReachabilityRewardFormula(CheckTask<storm::logic::EventuallyFormula, ConstantType> const& checkTask) override;
                virtual void specifyCumulativeRewardFormula(CheckTask<storm::logic::CumulativeRewardFormula, ConstantType> const& checkTask) override;
                
                virtual std::unique_ptr<CheckResult> computeQuantitativeValues(ParameterRegion<typename SparseModelType::ValueType> const& region, storm::solver::OptimizationDirection const& dirForParameters) override;
                

            private:
                storm::storage::BitVector maybeStates;
                std::vector<ConstantType> resultsForNonMaybeStates;
                boost::optional<uint_fast64_t> stepBound;
                
                std::unique_ptr<storm::transformer::ParameterLifter<typename SparseModelType::ValueType, ConstantType>> parameterLifter;
                std::unique_ptr<storm::solver::MinMaxLinearEquationSolverFactory<ConstantType>> solverFactory;
                
                // Results from the most recent solver call.
                boost::optional<storm::storage::TotalScheduler> minSched, maxSched;
                std::vector<ConstantType> x;
                boost::optional<ConstantType> lowerResultBound, upperResultBound;
            };
        }
    }
}
