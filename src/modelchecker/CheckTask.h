#ifndef STORM_MODELCHECKER_CHECKTASK_H_
#define STORM_MODELCHECKER_CHECKTASK_H_

#include <boost/optional.hpp>

#include "src/logic/Formulas.h"
#include "src/utility/constants.h"

#include "src/solver/OptimizationDirection.h"
#include "src/logic/ComparisonType.h"

namespace storm {
    namespace logic {
        class Formula;
    }
    
    namespace modelchecker {
        
        enum class CheckType {
            Probabilities, Rewards
        };
        
        /*
         * This class is used to customize the checking process of a formula.
         */
        template<typename FormulaType = storm::logic::Formula, typename ValueType = double>
        class CheckTask {
        public:
            template<typename OtherFormulaType, typename OtherValueType>
            friend class CheckTask;
            
            /*!
             * Creates a task object with the default options for the given formula.
             */
            CheckTask(FormulaType const& formula, bool onlyInitialStatesRelevant = false) : formula(formula) {
                this->onlyInitialStatesRelevant = onlyInitialStatesRelevant;
                this->produceSchedulers = false;
                this->qualitative = false;
                
                if (formula.isOperatorFormula()) {
                    storm::logic::OperatorFormula const& operatorFormula = formula.asOperatorFormula();
                    if (operatorFormula.hasOptimalityType()) {
                        this->optimizationDirection = operatorFormula.getOptimalityType();
                    }
                    
                    if (operatorFormula.hasBound()) {
                        if (onlyInitialStatesRelevant) {
                            this->bound = operatorFormula.getBound().convertToOtherValueType<ValueType>();
                        }

                        if (!optimizationDirection) {
                            this->optimizationDirection = operatorFormula.getComparisonType() == storm::logic::ComparisonType::Less || operatorFormula.getComparisonType() == storm::logic::ComparisonType::LessEqual ? OptimizationDirection::Maximize : OptimizationDirection::Minimize;
                        }
                    }
                }
                
                if (formula.isProbabilityOperatorFormula()) {
                    storm::logic::ProbabilityOperatorFormula const& probabilityOperatorFormula = formula.asProbabilityOperatorFormula();
                    
                    if (probabilityOperatorFormula.hasBound()) {
                        if (storm::utility::isZero(probabilityOperatorFormula.getThreshold()) || storm::utility::isOne(probabilityOperatorFormula.getThreshold())) {
                            this->qualitative = true;
                        }
                    }
                } else if (formula.isRewardOperatorFormula()) {
                    storm::logic::RewardOperatorFormula const& rewardOperatorFormula = formula.asRewardOperatorFormula();
                    this->rewardModel = rewardOperatorFormula.getOptionalRewardModelName();
                    
                    if (rewardOperatorFormula.hasBound()) {
                        if (storm::utility::isZero(rewardOperatorFormula.getThreshold())) {
                            this->qualitative = true;
                        }
                    }
                }
            }
            
            /*!
             * Copies the check task from the source while replacing the formula with the new one. In particular, this
             * changes the formula type of the check task object.
             */
            template<typename NewFormulaType>
            CheckTask<NewFormulaType, ValueType> substituteFormula(NewFormulaType const& newFormula) const {
                return CheckTask<NewFormulaType, ValueType>(newFormula, this->optimizationDirection, this->rewardModel, this->onlyInitialStatesRelevant, this->bound, this->qualitative, this->produceSchedulers, this->resultHint);
            }
            
            /*!
             * Retrieves the formula from this task.
             */
            FormulaType const& getFormula() const {
                return formula.get();
            }
            
            /*!
             * Retrieves whether an optimization direction was set.
             */
            bool isOptimizationDirectionSet() const {
                return static_cast<bool>(optimizationDirection);
            }
            
            /*!
             * Retrieves the optimization direction (if set).
             */
            storm::OptimizationDirection const& getOptimizationDirection() const {
                return optimizationDirection.get();
            }
            
            /*!
             * Retrieves whether a reward model was set.
             */
            bool isRewardModelSet() const {
                return static_cast<bool>(rewardModel);
            }
            
            /*!
             * Retrieves the reward model over which to perform the checking (if set).
             */
            std::string const& getRewardModel() const {
                return rewardModel.get();
            }
            
            /*!
             * Retrieves whether only the initial states are relevant in the computation.
             */
            bool isOnlyInitialStatesRelevantSet() const {
                return onlyInitialStatesRelevant;
            }
            
            /*!
             * Sets whether only initial states are relevant.
             */
            CheckTask<FormulaType, ValueType>& setOnlyInitialStatesRelevant(bool value = true) {
                this->onlyInitialStatesRelevant = value;
                return *this;
            }
            
            /*!
             * Retrieves whether there is a bound with which the values for the states will be compared.
             */
            bool isBoundSet() const {
                return static_cast<bool>(bound);
            }
            
            /*!
             * Retrieves the value of the bound (if set).
             */
            ValueType const& getBoundThreshold() const {
                return bound.get().threshold;
            }
            
            /*!
             * Retrieves the comparison type of the bound (if set).
             */
            storm::logic::ComparisonType const& getBoundComparisonType() const {
                return bound.get().comparisonType;
            }
            
            /*!
             * Retrieves the bound (if set).
             */
            storm::logic::Bound<ValueType> const& getBound() const {
                return bound.get();
            }
            
            /*!
             * Retrieves whether the computation only needs to be performed qualitatively, because the values will only
             * be compared to 0/1.
             */
            bool isQualitativeSet() const {
                return qualitative;
            }
            
            /*!
             * Sets whether to produce schedulers (if supported).
             */
            void setProduceSchedulers(bool produceSchedulers) {
                this->produceSchedulers = produceSchedulers;
            }
            
            /*!
             * Retrieves whether scheduler(s) are to be produced (if supported).
             */
            bool isProduceSchedulersSet() const {
                return produceSchedulers;
            }
            
            /*!
             * sets a vector that may serve as a hint for the (quantitative) model-checking result
             */
            void setResultHint(std::vector<ValueType> const& hint){
                this->resultHint = hint;
            }
            
            /*!
             * sets a vector that may serve as a hint for the (quantitative) model-checking result
             */
            void setResultHint(std::vector<ValueType>&& hint){
                this->resultHint = hint;
            }
            
            /*!
             * Retrieves whether there is a vector that may serve as a hint for the (quantitative) model-checking result
             */
            bool isResultHintSet() const {
                return static_cast<bool>(resultHint);
            }
            
            /*!
             * Retrieves the vector that may serve as a hint for the (quantitative) model-checking result
             */
            std::vector<ValueType> const& getResultVectorHint() const {
                return resultHint.get();
            }
            boost::optional<std::vector<ValueType>> const& getOptionalResultVectorHint() const{
                return resultHint;
            }
            
        private:
            /*!
             * Creates a task object with the given options.
             *
             * @param formula The formula to attach to the task.
             * @param optimizationDirection If set, the probabilities will be minimized/maximized.
             * @param rewardModelName If given, the checking has to be done wrt. to this reward model.
             * @param onlyInitialStatesRelevant If set to true, the model checker may decide to only compute the values
             * for the initial states.
             * @param bound The bound with which the states will be compared.
             * @param qualitative A flag specifying whether the property needs to be checked qualitatively, i.e. compared
             * with bounds 0/1.
             * @param produceSchedulers If supported by the model checker and the model formalism, schedulers to achieve
             * a value will be produced if this flag is set.
             */
            CheckTask(std::reference_wrapper<FormulaType const> const& formula, boost::optional<storm::OptimizationDirection> const& optimizationDirection, boost::optional<std::string> const& rewardModel, bool onlyInitialStatesRelevant, boost::optional<storm::logic::Bound<ValueType>> const& bound, bool qualitative, bool produceSchedulers, boost::optional<std::vector<ValueType>> const& resultHint) : formula(formula), optimizationDirection(optimizationDirection), rewardModel(rewardModel), onlyInitialStatesRelevant(onlyInitialStatesRelevant), bound(bound), qualitative(qualitative), produceSchedulers(produceSchedulers), resultHint(resultHint) {
                // Intentionally left empty.
            }
            
            // The formula that is to be checked.
            std::reference_wrapper<FormulaType const> formula;
            
            // If set, the probabilities will be minimized/maximized.
            boost::optional<storm::OptimizationDirection> optimizationDirection;

            // If set, the reward property has to be interpreted over this model.
            boost::optional<std::string> rewardModel;
            
            // If set to true, the model checker may decide to only compute the values for the initial states.
            bool onlyInitialStatesRelevant;

            // The bound with which the states will be compared.
            boost::optional<storm::logic::Bound<ValueType>> bound;
            
            // A flag specifying whether the property needs to be checked qualitatively, i.e. compared with bounds 0/1.
            bool qualitative;
            
            // If supported by the model checker and the model formalism, schedulers to achieve a value will be produced
            // if this flag is set.
            bool produceSchedulers;
            
            // If supported by the model checker and the model formalism, this vector serves as initial guess for the
            // solution.
            boost::optional<std::vector<ValueType>> resultHint;
       //     boost::optional<storm::storage::Scheduler> schedulerHint; //TODO: will need two schedulers for games
        };
        
    }
}

#endif /* STORM_MODELCHECKER_CHECKTASK_H_ */
