#include "src/storm/modelchecker/multiobjective/pcaa/SparsePcaaQuantitativeQuery.h"

#include "src/storm/adapters/CarlAdapter.h"
#include "src/storm/models/sparse/Mdp.h"
#include "src/storm/models/sparse/MarkovAutomaton.h"
#include "src/storm/models/sparse/StandardRewardModel.h"
#include "src/storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "src/storm/modelchecker/results/ExplicitQuantitativeCheckResult.h"
#include "src/storm/utility/constants.h"
#include "src/storm/utility/vector.h"
#include "src/storm/settings//SettingsManager.h"
#include "src/storm/settings/modules/MultiObjectiveSettings.h"
#include "src/storm/settings/modules/GeneralSettings.h"

namespace storm {
    namespace modelchecker {
        namespace multiobjective {
            
            
            template <class SparseModelType, typename GeometryValueType>
            SparsePcaaQuantitativeQuery<SparseModelType, GeometryValueType>::SparsePcaaQuantitativeQuery(SparsePcaaPreprocessorReturnType<SparseModelType>& preprocessorResult) : SparsePcaaQuery<SparseModelType, GeometryValueType>(preprocessorResult) {
                STORM_LOG_ASSERT(preprocessorResult.queryType==SparsePcaaPreprocessorReturnType<SparseModelType>::QueryType::Quantitative, "Invalid query Type");
                STORM_LOG_ASSERT(preprocessorResult.indexOfOptimizingObjective, "Detected quantitative query but index of optimizing objective is not set.");
                indexOfOptimizingObjective = *preprocessorResult.indexOfOptimizingObjective;
                initializeThresholdData();
                
                // Set the maximum distance between lower and upper bound of the weightVectorChecker result.
                this->weightVectorChecker->setWeightedPrecision(storm::utility::convertNumber<typename SparseModelType::ValueType>(0.1));
            }
            
            template <class SparseModelType, typename GeometryValueType>
            void SparsePcaaQuantitativeQuery<SparseModelType, GeometryValueType>::initializeThresholdData() {
                thresholds.reserve(this->objectives.size());
                strictThresholds = storm::storage::BitVector(this->objectives.size(), false);
                std::vector<storm::storage::geometry::Halfspace<GeometryValueType>> thresholdConstraints;
                thresholdConstraints.reserve(this->objectives.size()-1);
                for(uint_fast64_t objIndex = 0; objIndex < this->objectives.size(); ++objIndex) {
                    if(this->objectives[objIndex].threshold) {
                        thresholds.push_back(storm::utility::convertNumber<GeometryValueType>(*this->objectives[objIndex].threshold));
                        WeightVector normalVector(this->objectives.size(), storm::utility::zero<GeometryValueType>());
                        normalVector[objIndex] = -storm::utility::one<GeometryValueType>();
                        thresholdConstraints.emplace_back(std::move(normalVector), -thresholds.back());
                        strictThresholds.set(objIndex, this->objectives[objIndex].thresholdIsStrict);
                    } else {
                        thresholds.push_back(storm::utility::zero<GeometryValueType>());
                    }
                }
                // Note: If we have a single objective (i.e., no objectives with thresholds), thresholdsAsPolytope gets no constraints
                thresholdsAsPolytope = storm::storage::geometry::Polytope<GeometryValueType>::create(thresholdConstraints);
            }
            
            template <class SparseModelType, typename GeometryValueType>
            std::unique_ptr<CheckResult> SparsePcaaQuantitativeQuery<SparseModelType, GeometryValueType>::check() {
               
                
                // First find one solution that achieves the given thresholds ...
                if(this->checkAchievability()) {
                    // ... then improve it
                    GeometryValueType result = this->improveSolution();
                    
                    // transform the obtained result for the preprocessed model to a result w.r.t. the original model and return the checkresult
                    typename SparseModelType::ValueType resultForOriginalModel =
                            storm::utility::convertNumber<typename SparseModelType::ValueType>(result) *
                            this->objectives[indexOfOptimizingObjective].toOriginalValueTransformationFactor +
                            this->objectives[indexOfOptimizingObjective].toOriginalValueTransformationOffset;
                    return std::unique_ptr<CheckResult>(new ExplicitQuantitativeCheckResult<typename SparseModelType::ValueType>(this->originalModel.getInitialStates().getNextSetIndex(0), resultForOriginalModel));
                } else {
                    return std::unique_ptr<CheckResult>(new ExplicitQualitativeCheckResult(this->originalModel.getInitialStates().getNextSetIndex(0), false));
                }
     
                
            }
            
            template <class SparseModelType, typename GeometryValueType>
            bool SparsePcaaQuantitativeQuery<SparseModelType, GeometryValueType>::checkAchievability() {
                // We don't care for the optimizing objective at this point
                this->diracWeightVectorsToBeChecked.set(indexOfOptimizingObjective, false);
                
                while(!this->maxStepsPerformed()){
                    WeightVector separatingVector = this->findSeparatingVector(thresholds);
                    this->updateWeightedPrecisionInAchievabilityPhase(separatingVector);
                    this->performRefinementStep(std::move(separatingVector));
                    //Pick the threshold for the optimizing objective low enough so valid solutions are not excluded
                    thresholds[indexOfOptimizingObjective] = std::min(thresholds[indexOfOptimizingObjective], this->refinementSteps.back().lowerBoundPoint[indexOfOptimizingObjective]);
                    if(!checkIfThresholdsAreSatisfied(this->overApproximation)){
                        return false;
                    }
                    if(checkIfThresholdsAreSatisfied(this->underApproximation)){
                        return true;
                    }
                }
                STORM_LOG_ERROR("Could not check whether thresholds are achievable: Exceeded maximum number of refinement steps");
                return false;
            }
            
            template <class SparseModelType, typename GeometryValueType>
            void SparsePcaaQuantitativeQuery<SparseModelType, GeometryValueType>::updateWeightedPrecisionInAchievabilityPhase(WeightVector const& weights) {
                // Our heuristic considers the distance between the under- and the over approximation w.r.t. the given direction
                std::pair<Point, bool> optimizationResOverApprox = this->overApproximation->optimize(weights);
                if(optimizationResOverApprox.second) {
                    std::pair<Point, bool> optimizationResUnderApprox = this->underApproximation->optimize(weights);
                    if(optimizationResUnderApprox.second) {
                        GeometryValueType distance = storm::utility::vector::dotProduct(optimizationResOverApprox.first, weights) - storm::utility::vector::dotProduct(optimizationResUnderApprox.first, weights);
                        STORM_LOG_ASSERT(distance >= storm::utility::zero<GeometryValueType>(), "Negative distance between under- and over approximation was not expected");
                        // Normalize the distance by dividing it with the Euclidean Norm of the weight-vector
                        distance /= storm::utility::sqrt(storm::utility::vector::dotProduct(weights, weights));
                        distance /= GeometryValueType(2);
                        this->weightVectorChecker->setWeightedPrecision(storm::utility::convertNumber<typename SparseModelType::ValueType>(distance));
                    }
                }
                // do not update the precision if one of the approximations is unbounded in the provided direction
            }

            template <class SparseModelType, typename GeometryValueType>
            GeometryValueType SparsePcaaQuantitativeQuery<SparseModelType, GeometryValueType>::improveSolution() {
                this->diracWeightVectorsToBeChecked.clear(); // Only check weight vectors that can actually improve the solution
            
                WeightVector directionOfOptimizingObjective(this->objectives.size(), storm::utility::zero<GeometryValueType>());
                directionOfOptimizingObjective[indexOfOptimizingObjective] = storm::utility::one<GeometryValueType>();
            
                // Improve the found solution.
                // Note that we do not care whether a threshold is strict anymore, because the resulting optimum should be
                // the supremum over all strategies. Hence, one could combine a scheduler inducing the optimum value (but possibly violating strict
                // thresholds) and (with very low probability) a scheduler that satisfies all (possibly strict) thresholds.
                GeometryValueType result = storm::utility::zero<GeometryValueType>();
                while(!this->maxStepsPerformed()) {
                    std::pair<Point, bool> optimizationRes = this->underApproximation->intersection(thresholdsAsPolytope)->optimize(directionOfOptimizingObjective);
                    STORM_LOG_THROW(optimizationRes.second, storm::exceptions::UnexpectedException, "The underapproximation is either unbounded or empty.");
                    result = optimizationRes.first[indexOfOptimizingObjective];
                    STORM_LOG_DEBUG("Best solution found so far is ~" << storm::utility::convertNumber<double>(result) << ".");
                    //Compute an upper bound for the optimum and check for convergence
                    optimizationRes = this->overApproximation->intersection(thresholdsAsPolytope)->optimize(directionOfOptimizingObjective);
                    if(optimizationRes.second) {
                        GeometryValueType precisionOfResult = optimizationRes.first[indexOfOptimizingObjective] - result;
                        if(precisionOfResult < storm::utility::convertNumber<GeometryValueType>(storm::settings::getModule<storm::settings::modules::MultiObjectiveSettings>().getPrecision())) {
                            // Goal precision reached!
                            return result;
                        } else {
                            STORM_LOG_DEBUG("Solution can be improved by at most " << storm::utility::convertNumber<double>(precisionOfResult));
                            thresholds[indexOfOptimizingObjective] = optimizationRes.first[indexOfOptimizingObjective];
                        }
                    } else {
                        thresholds[indexOfOptimizingObjective] = result + storm::utility::one<GeometryValueType>();
                    }
                    WeightVector separatingVector = this->findSeparatingVector(thresholds);
                    this->updateWeightedPrecisionInImprovingPhase(separatingVector);
                    this->performRefinementStep(std::move(separatingVector));
                }
               STORM_LOG_ERROR("Could not reach the desired precision: Exceeded maximum number of refinement steps");
               return result;
            }
            
            
            template <class SparseModelType, typename GeometryValueType>
            void SparsePcaaQuantitativeQuery<SparseModelType, GeometryValueType>::updateWeightedPrecisionInImprovingPhase(WeightVector const& weights) {
                STORM_LOG_THROW(!storm::utility::isZero(weights[this->indexOfOptimizingObjective]), exceptions::UnexpectedException, "The chosen weight-vector gives zero weight for the objective that is to be optimized.");
                // If weighs[indexOfOptimizingObjective] is low, the computation of the weightVectorChecker needs to be more precise.
                // Our heuristic ensures that if p is the new vertex of the under-approximation, then max{ eps | p' = p + (0..0 eps 0..0) is in the over-approximation } <= multiobjective_precision/2
                    GeometryValueType weightedPrecision = weights[this->indexOfOptimizingObjective] * storm::utility::convertNumber<GeometryValueType>(storm::settings::getModule<storm::settings::modules::MultiObjectiveSettings>().getPrecision());
                    // Normalize by division with the Euclidean Norm of the weight-vector
                    weightedPrecision /= storm::utility::sqrt(storm::utility::vector::dotProduct(weights, weights));
                    weightedPrecision /= GeometryValueType(2);
                    this->weightVectorChecker->setWeightedPrecision(storm::utility::convertNumber<typename SparseModelType::ValueType>(weightedPrecision));
            }
            
            
            template <class SparseModelType, typename GeometryValueType>
            bool SparsePcaaQuantitativeQuery<SparseModelType, GeometryValueType>::checkIfThresholdsAreSatisfied(std::shared_ptr<storm::storage::geometry::Polytope<GeometryValueType>> const& polytope) {
                std::vector<storm::storage::geometry::Halfspace<GeometryValueType>> halfspaces = polytope->getHalfspaces();
                for(auto const& h : halfspaces) {
                    if(storm::utility::isZero(h.distance(thresholds))) {
                        // Check if the threshold point is on the boundary of the halfspace and whether this is violates strict thresholds
                        if(h.isPointOnBoundary(thresholds)) {
                            for(auto strictThreshold : strictThresholds) {
                                if(h.normalVector()[strictThreshold] > storm::utility::zero<GeometryValueType>()) {
                                    return false;
                                }
                            }
                        }
                    } else {
                        return false;
                    }
                }
                return true;
            }
            

            
#ifdef STORM_HAVE_CARL
            template class SparsePcaaQuantitativeQuery<storm::models::sparse::Mdp<double>, storm::RationalNumber>;
            template class SparsePcaaQuantitativeQuery<storm::models::sparse::MarkovAutomaton<double>, storm::RationalNumber>;
            
            template class SparsePcaaQuantitativeQuery<storm::models::sparse::Mdp<storm::RationalNumber>, storm::RationalNumber>;
          //  template class SparsePcaaQuantitativeQuery<storm::models::sparse::MarkovAutomaton<storm::RationalNumber>, storm::RationalNumber>;
#endif
        }
    }
}
