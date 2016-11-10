#include "src/modelchecker/multiobjective/pcaa/SparsePcaaQuery.h"

#include "src/adapters/CarlAdapter.h"
#include "src/models/sparse/Mdp.h"
#include "src/models/sparse/MarkovAutomaton.h"
#include "src/models/sparse/StandardRewardModel.h"
#include "src/modelchecker/multiobjective/pcaa/PcaaObjective.h"
#include "src/modelchecker/multiobjective/pcaa/SparseMdpPcaaWeightVectorChecker.h"
#include "src/modelchecker/multiobjective/pcaa/SparseMaPcaaWeightVectorChecker.h"
#include "src/settings//SettingsManager.h"
#include "src/settings/modules/MultiObjectiveSettings.h"
#include "src/storage/geometry/Hyperrectangle.h"
#include "src/utility/constants.h"
#include "src/utility/vector.h"
#include "src/utility/export.h"

#include "src/exceptions/UnexpectedException.h"

namespace storm {
    namespace modelchecker {
        namespace multiobjective {
            
            
            template <class SparseModelType, typename GeometryValueType>
            SparsePcaaQuery<SparseModelType, GeometryValueType>::SparsePcaaQuery(SparsePcaaPreprocessorReturnType<SparseModelType>& preprocessorResult) :
                originalModel(preprocessorResult.originalModel), originalFormula(preprocessorResult.originalFormula),
                preprocessedModel(std::move(preprocessorResult.preprocessedModel)), objectives(std::move(preprocessorResult.objectives)) {
                initializeWeightVectorChecker(preprocessedModel, objectives, preprocessorResult.actionsWithNegativeReward, preprocessorResult.ecActions, preprocessorResult.possiblyRecurrentStates);
                this->diracWeightVectorsToBeChecked = storm::storage::BitVector(this->objectives.size(), true);
                this->overApproximation = storm::storage::geometry::Polytope<GeometryValueType>::createUniversalPolytope();
                this->underApproximation = storm::storage::geometry::Polytope<GeometryValueType>::createEmptyPolytope();
            }
            
            template<>
            void SparsePcaaQuery<storm::models::sparse::Mdp<double>, storm::RationalNumber>::initializeWeightVectorChecker(storm::models::sparse::Mdp<double> const& model, std::vector<PcaaObjective<double>> const& objectives, storm::storage::BitVector const& actionsWithNegativeReward, storm::storage::BitVector const& ecActions, storm::storage::BitVector const& possiblyRecurrentStates) {
                this->weightVectorChecker = std::unique_ptr<SparsePcaaWeightVectorChecker<storm::models::sparse::Mdp<double>>>(new SparseMdpPcaaWeightVectorChecker<storm::models::sparse::Mdp<double>>(model, objectives, actionsWithNegativeReward, ecActions, possiblyRecurrentStates));
            }
            
            template<>
            void SparsePcaaQuery<storm::models::sparse::Mdp<storm::RationalNumber>, storm::RationalNumber>::initializeWeightVectorChecker(storm::models::sparse::Mdp<storm::RationalNumber> const& model, std::vector<PcaaObjective<storm::RationalNumber>> const& objectives, storm::storage::BitVector const& actionsWithNegativeReward, storm::storage::BitVector const& ecActions, storm::storage::BitVector const& possiblyRecurrentStates) {
                this->weightVectorChecker = std::unique_ptr<SparsePcaaWeightVectorChecker<storm::models::sparse::Mdp<storm::RationalNumber>>>(new SparseMdpPcaaWeightVectorChecker<storm::models::sparse::Mdp<storm::RationalNumber>>(model, objectives, actionsWithNegativeReward, ecActions, possiblyRecurrentStates));
            }
            
            template<>
            void SparsePcaaQuery<storm::models::sparse::MarkovAutomaton<double>, storm::RationalNumber>::initializeWeightVectorChecker(storm::models::sparse::MarkovAutomaton<double> const& model, std::vector<PcaaObjective<double>> const& objectives, storm::storage::BitVector const& actionsWithNegativeReward, storm::storage::BitVector const& ecActions, storm::storage::BitVector const& possiblyRecurrentStates) {
                this->weightVectorChecker = std::unique_ptr<SparsePcaaWeightVectorChecker<storm::models::sparse::MarkovAutomaton<double>>>(new SparseMaPcaaWeightVectorChecker<storm::models::sparse::MarkovAutomaton<double>>(model, objectives, actionsWithNegativeReward, ecActions, possiblyRecurrentStates));
            }
            
            template <class SparseModelType, typename GeometryValueType>
            typename SparsePcaaQuery<SparseModelType, GeometryValueType>::WeightVector SparsePcaaQuery<SparseModelType, GeometryValueType>::findSeparatingVector(Point const& pointToBeSeparated) {
                STORM_LOG_DEBUG("Searching a weight vector to seperate the point given by " << storm::utility::vector::toString(storm::utility::vector::convertNumericVector<double>(pointToBeSeparated)) << ".");
                
                if(underApproximation->isEmpty()) {
                    // In this case, every weight vector is separating
                    uint_fast64_t objIndex = diracWeightVectorsToBeChecked.getNextSetIndex(0) % pointToBeSeparated.size();
                    WeightVector result(pointToBeSeparated.size(), storm::utility::zero<GeometryValueType>());
                    result[objIndex] = storm::utility::one<GeometryValueType>();
                    diracWeightVectorsToBeChecked.set(objIndex, false);
                    return result;
                }
                
                // Reaching this point means that the underApproximation contains halfspaces. The seperating vector has to be the normal vector of one of these halfspaces.
                // We pick one with maximal distance to the given point. However, Dirac weight vectors that only assign a non-zero weight to a single objective take precedence.
                std::vector<storm::storage::geometry::Halfspace<GeometryValueType>> halfspaces = underApproximation->getHalfspaces();
                uint_fast64_t farestHalfspaceIndex = halfspaces.size();
                GeometryValueType farestDistance = -storm::utility::one<GeometryValueType>();
                bool foundSeparatingDiracVector = false;
                for(uint_fast64_t halfspaceIndex = 0; halfspaceIndex < halfspaces.size(); ++halfspaceIndex) {
                    GeometryValueType distance = -halfspaces[halfspaceIndex].euclideanDistance(pointToBeSeparated);
                    if(distance >= storm::utility::zero<GeometryValueType>()) {
                        storm::storage::BitVector nonZeroVectorEntries = ~storm::utility::vector::filterZero<GeometryValueType>(halfspaces[halfspaceIndex].normalVector());
                        bool isSingleObjectiveVector = nonZeroVectorEntries.getNumberOfSetBits() == 1 && diracWeightVectorsToBeChecked.get(nonZeroVectorEntries.getNextSetIndex(0));
                        // Check if this halfspace is a better candidate than the current one
                        if((!foundSeparatingDiracVector && isSingleObjectiveVector ) || (foundSeparatingDiracVector==isSingleObjectiveVector && distance>farestDistance)) {
                            foundSeparatingDiracVector = foundSeparatingDiracVector || isSingleObjectiveVector;
                            farestHalfspaceIndex = halfspaceIndex;
                            farestDistance = distance;
                        }
                    }
                }
                if(foundSeparatingDiracVector) {
                    diracWeightVectorsToBeChecked &= storm::utility::vector::filterZero<GeometryValueType>(halfspaces[farestHalfspaceIndex].normalVector());
                }
                
                STORM_LOG_THROW(farestHalfspaceIndex<halfspaces.size(), storm::exceptions::UnexpectedException, "There is no seperating vector.");
                STORM_LOG_DEBUG("Found separating  weight vector: " << storm::utility::vector::toString(storm::utility::vector::convertNumericVector<double>(halfspaces[farestHalfspaceIndex].normalVector())) << ".");
                return halfspaces[farestHalfspaceIndex].normalVector();
            }
            
            template <class SparseModelType, typename GeometryValueType>
            void SparsePcaaQuery<SparseModelType, GeometryValueType>::performRefinementStep(WeightVector&& direction) {
                // Normalize the direction vector so that the entries sum up to one
                storm::utility::vector::scaleVectorInPlace(direction, storm::utility::one<GeometryValueType>() / std::accumulate(direction.begin(), direction.end(), storm::utility::zero<GeometryValueType>()));
                // Check if we already did a refinement step with that direction vector. If this is the case, we increase the precision.
                // We start with the most recent steps to consider the most recent result for this direction vector
                boost::optional<typename SparseModelType::ValueType> oldMaximumLowerUpperBoundGap;
                for(auto stepIt = refinementSteps.rbegin(); stepIt != refinementSteps.rend(); ++stepIt) {
                    if(stepIt->weightVector == direction) {
                        STORM_LOG_WARN("Performing multiple refinement steps with the same direction vector.");
                        oldMaximumLowerUpperBoundGap = weightVectorChecker->getMaximumLowerUpperBoundGap();
                        std::vector<GeometryValueType> lowerUpperDistances = stepIt->upperBoundPoint;
                        storm::utility::vector::subtractVectors(lowerUpperDistances, stepIt->lowerBoundPoint, lowerUpperDistances);
                        // shorten the distance between lower and upper bound for the new result by multiplying the current distance with 0.5
                        // TODO: try other values/strategies?
                        GeometryValueType distance = storm::utility::sqrt(storm::utility::vector::dotProduct(lowerUpperDistances, lowerUpperDistances));
                        weightVectorChecker->setMaximumLowerUpperBoundGap(std::min(*oldMaximumLowerUpperBoundGap, storm::utility::convertNumber<typename SparseModelType::ValueType>(distance) * storm::utility::convertNumber<typename SparseModelType::ValueType>(0.5)));
                        break;
                    }
                }
                weightVectorChecker->check(storm::utility::vector::convertNumericVector<typename SparseModelType::ValueType>(direction));
                if(oldMaximumLowerUpperBoundGap) {
                    // Reset the precision back to the previous values
                    weightVectorChecker->setMaximumLowerUpperBoundGap(*oldMaximumLowerUpperBoundGap);
                }
                STORM_LOG_DEBUG("weighted objectives checker result (lower bounds) is " << storm::utility::vector::toString(storm::utility::vector::convertNumericVector<double>(weightVectorChecker->getLowerBoundsOfInitialStateResults())));
                RefinementStep step;
                step.weightVector = direction;
                step.lowerBoundPoint = storm::utility::vector::convertNumericVector<GeometryValueType>(weightVectorChecker->getLowerBoundsOfInitialStateResults());
                step.upperBoundPoint = storm::utility::vector::convertNumericVector<GeometryValueType>(weightVectorChecker->getUpperBoundsOfInitialStateResults());
                refinementSteps.push_back(std::move(step));
                
                updateOverApproximation();
                updateUnderApproximation();
            }
            
            template <class SparseModelType, typename GeometryValueType>
            void SparsePcaaQuery<SparseModelType, GeometryValueType>::updateOverApproximation() {
                storm::storage::geometry::Halfspace<GeometryValueType> h(refinementSteps.back().weightVector, storm::utility::vector::dotProduct(refinementSteps.back().weightVector, refinementSteps.back().upperBoundPoint));
                
                // Due to numerical issues, it might be the case that the updated overapproximation does not contain the underapproximation,
                // e.g., when the new point is strictly contained in the underapproximation. Check if this is the case.
                GeometryValueType maximumOffset = h.offset();
                for(auto const& step : refinementSteps){
                    maximumOffset = std::max(maximumOffset, storm::utility::vector::dotProduct(h.normalVector(), step.lowerBoundPoint));
                }
                if(maximumOffset > h.offset()){
                    // We correct the issue by shifting the halfspace such that it contains the underapproximation
                    h.offset() = maximumOffset;
                    STORM_LOG_WARN("Numerical issues: The overapproximation would not contain the underapproximation. Hence, a halfspace is shifted by " << storm::utility::convertNumber<double>(h.euclideanDistance(refinementSteps.back().upperBoundPoint)) << ".");
                }
                overApproximation = overApproximation->intersection(h);
                STORM_LOG_DEBUG("Updated OverApproximation to " << overApproximation->toString(true));
            }
            
            template <class SparseModelType, typename GeometryValueType>
            void SparsePcaaQuery<SparseModelType, GeometryValueType>::updateUnderApproximation() {
                std::vector<Point> paretoPoints;
                paretoPoints.reserve(refinementSteps.size());
                for(auto const& step : refinementSteps) {
                    paretoPoints.push_back(step.lowerBoundPoint);
                }
                underApproximation = storm::storage::geometry::Polytope<GeometryValueType>::createDownwardClosure(paretoPoints);
                STORM_LOG_DEBUG("Updated UnderApproximation to " << underApproximation->toString(true));
            }
            
            template <class SparseModelType, typename GeometryValueType>
            bool SparsePcaaQuery<SparseModelType, GeometryValueType>::maxStepsPerformed() const {
                return storm::settings::getModule<storm::settings::modules::MultiObjectiveSettings>().isMaxStepsSet() &&
                this->refinementSteps.size() >= storm::settings::getModule<storm::settings::modules::MultiObjectiveSettings>().getMaxSteps();
            }
            
            
            template<typename SparseModelType, typename GeometryValueType>
            typename SparsePcaaQuery<SparseModelType, GeometryValueType>::Point SparsePcaaQuery<SparseModelType, GeometryValueType>::transformPointToOriginalModel(Point const& point) const {
                Point result;
                result.reserve(point.size());
                for(uint_fast64_t objIndex = 0; objIndex < this->objectives.size(); ++objIndex) {
                    result.push_back(point[objIndex] * storm::utility::convertNumber<GeometryValueType>(this->objectives[objIndex].toOriginalValueTransformationFactor) + storm::utility::convertNumber<GeometryValueType>(this->objectives[objIndex].toOriginalValueTransformationOffset));
                }
                return result;
            }
            
            template<typename SparseModelType, typename GeometryValueType>
            std::shared_ptr<storm::storage::geometry::Polytope<GeometryValueType>> SparsePcaaQuery<SparseModelType, GeometryValueType>::transformPolytopeToOriginalModel(std::shared_ptr<storm::storage::geometry::Polytope<GeometryValueType>> const& polytope) const {
                if(polytope->isEmpty()) {
                    return storm::storage::geometry::Polytope<GeometryValueType>::createEmptyPolytope();
                }
                if(polytope->isUniversal()) {
                    return storm::storage::geometry::Polytope<GeometryValueType>::createUniversalPolytope();
                }
                uint_fast64_t numObjectives = this->objectives.size();
                std::vector<std::vector<GeometryValueType>> transformationMatrix(numObjectives, std::vector<GeometryValueType>(numObjectives, storm::utility::zero<GeometryValueType>()));
                std::vector<GeometryValueType> transformationVector;
                transformationVector.reserve(numObjectives);
                for(uint_fast64_t objIndex = 0; objIndex < numObjectives; ++objIndex) {
                    transformationMatrix[objIndex][objIndex] = storm::utility::convertNumber<GeometryValueType>(this->objectives[objIndex].toOriginalValueTransformationFactor);
                    transformationVector.push_back(storm::utility::convertNumber<GeometryValueType>(this->objectives[objIndex].toOriginalValueTransformationOffset));
                }
                return polytope->affineTransformation(transformationMatrix, transformationVector);
            }
            
            template<typename SparseModelType, typename GeometryValueType>
            void SparsePcaaQuery<SparseModelType, GeometryValueType>::exportPlotOfCurrentApproximation(std::string const& destinationDir) const {
               
                STORM_LOG_ERROR_COND(objectives.size()==2, "Exporting plot requested but this is only implemented for the two-dimensional case.");
                
                auto transformedUnderApprox = transformPolytopeToOriginalModel(underApproximation);
                auto transformedOverApprox = transformPolytopeToOriginalModel(overApproximation);
                
                // Get pareto points as well as a hyperrectangle that is used to guarantee that the resulting polytopes are bounded.
                storm::storage::geometry::Hyperrectangle<GeometryValueType> boundaries(std::vector<GeometryValueType>(objectives.size(), storm::utility::zero<GeometryValueType>()), std::vector<GeometryValueType>(objectives.size(), storm::utility::zero<GeometryValueType>()));
                std::vector<std::vector<GeometryValueType>> paretoPoints;
                paretoPoints.reserve(refinementSteps.size());
                for(auto const& step : refinementSteps) {
                    paretoPoints.push_back(transformPointToOriginalModel(step.lowerBoundPoint));
                    boundaries.enlarge(paretoPoints.back());
                }
                auto underApproxVertices = transformedUnderApprox->getVertices();
                for(auto const& v : underApproxVertices) {
                    boundaries.enlarge(v);
                }
                auto overApproxVertices = transformedOverApprox->getVertices();
                for(auto const& v : overApproxVertices) {
                    boundaries.enlarge(v);
                }
                
                //Further enlarge the boundaries a little
                storm::utility::vector::scaleVectorInPlace(boundaries.lowerBounds(), GeometryValueType(15) / GeometryValueType(10));
                storm::utility::vector::scaleVectorInPlace(boundaries.upperBounds(), GeometryValueType(15) / GeometryValueType(10));
                
                auto boundariesAsPolytope = boundaries.asPolytope();
                std::vector<std::string> columnHeaders = {"x", "y"};
                
                std::vector<std::vector<double>> pointsForPlotting;
                underApproxVertices = transformedUnderApprox->intersection(boundariesAsPolytope)->getVerticesInClockwiseOrder();
                pointsForPlotting.reserve(underApproxVertices.size());
                for(auto const& v : underApproxVertices) {
                    pointsForPlotting.push_back(storm::utility::vector::convertNumericVector<double>(v));
                }
                storm::utility::exportDataToCSVFile(destinationDir + "underapproximation.csv", pointsForPlotting, columnHeaders);
                
                pointsForPlotting.clear();
                overApproxVertices = transformedOverApprox->intersection(boundariesAsPolytope)->getVerticesInClockwiseOrder();
                pointsForPlotting.reserve(overApproxVertices.size());
                for(auto const& v : overApproxVertices) {
                    pointsForPlotting.push_back(storm::utility::vector::convertNumericVector<double>(v));
                }
                storm::utility::exportDataToCSVFile(destinationDir + "overapproximation.csv", pointsForPlotting, columnHeaders);
                
                pointsForPlotting.clear();
                pointsForPlotting.reserve(paretoPoints.size());
                for(auto const& v : paretoPoints) {
                    pointsForPlotting.push_back(storm::utility::vector::convertNumericVector<double>(v));
                }
                storm::utility::exportDataToCSVFile(destinationDir + "paretopoints.csv", pointsForPlotting, columnHeaders);
                
                pointsForPlotting.clear();
                auto boundVertices = boundariesAsPolytope->getVerticesInClockwiseOrder();
                pointsForPlotting.reserve(4);
                for(auto const& v : boundVertices) {
                    pointsForPlotting.push_back(storm::utility::vector::convertNumericVector<double>(v));
                }
                storm::utility::exportDataToCSVFile(destinationDir + "boundaries.csv", pointsForPlotting, columnHeaders);
            }
            
#ifdef STORM_HAVE_CARL
            template class SparsePcaaQuery<storm::models::sparse::Mdp<double>, storm::RationalNumber>;
            template class SparsePcaaQuery<storm::models::sparse::MarkovAutomaton<double>, storm::RationalNumber>;
            
            template class SparsePcaaQuery<storm::models::sparse::Mdp<storm::RationalNumber>, storm::RationalNumber>;
#endif
        }
    }
}
