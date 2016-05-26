#include "src/logic/FragmentSpecification.h"

#include <iostream>

namespace storm {
    namespace logic {
        
        FragmentSpecification propositional() {
            FragmentSpecification propositional;
            
            propositional.setBooleanLiteralFormulasAllowed(true);
            propositional.setBinaryBooleanStateFormulasAllowed(true);
            propositional.setUnaryBooleanStateFormulasAllowed(true);
            propositional.setAtomicExpressionFormulasAllowed(true);
            propositional.setAtomicLabelFormulasAllowed(true);
            
            return propositional;
        }
        
        FragmentSpecification reachability() {
            FragmentSpecification reachability = propositional();
            
            reachability.setProbabilityOperatorsAllowed(true);
            reachability.setUntilFormulasAllowed(true);
            reachability.setReachabilityProbabilityFormulasAllowed(true);
            reachability.setOperatorAtTopLevelRequired(true);
            reachability.setNestedOperatorsAllowed(false);
            
            return reachability;
        }
        
        FragmentSpecification pctl() {
            FragmentSpecification pctl = propositional();
            
            pctl.setProbabilityOperatorsAllowed(true);
            pctl.setGloballyFormulasAllowed(true);
            pctl.setReachabilityProbabilityFormulasAllowed(true);
            pctl.setNextFormulasAllowed(true);
            pctl.setUntilFormulasAllowed(true);
            pctl.setBoundedUntilFormulasAllowed(true);
            pctl.setStepBoundedUntilFormulasAllowed(true);
            
            return pctl;
        }
        
        FragmentSpecification flatPctl() {
            FragmentSpecification flatPctl = pctl();
            
            flatPctl.setNestedOperatorsAllowed(false);
            
            return flatPctl;
        }
        
        FragmentSpecification prctl() {
            FragmentSpecification prctl = pctl();
            
            prctl.setRewardOperatorsAllowed(true);
            prctl.setCumulativeRewardFormulasAllowed(true);
            prctl.setInstantaneousFormulasAllowed(true);
            prctl.setReachabilityRewardFormulasAllowed(true);
            prctl.setLongRunAverageOperatorsAllowed(true);
            
            return prctl;
        }
        
        FragmentSpecification csl() {
            FragmentSpecification csl = pctl();
            
            csl.setTimeBoundedUntilFormulasAllowed(true);
            
            return csl;
        }
        
        FragmentSpecification csrl() {
            FragmentSpecification csrl = csl();
            
            csrl.setRewardOperatorsAllowed(true);
            csrl.setCumulativeRewardFormulasAllowed(true);
            csrl.setInstantaneousFormulasAllowed(true);
            csrl.setReachabilityRewardFormulasAllowed(true);
            csrl.setLongRunAverageOperatorsAllowed(true);
            
            return csrl;
        }
        
        FragmentSpecification multiObjective() {
            FragmentSpecification multiObjective = propositional();
            
            multiObjective.setMultiObjectiveFormulasAllowed(true);
            multiObjective.setProbabilityOperatorsAllowed(true);
            multiObjective.setUntilFormulasAllowed(true);
            multiObjective.setReachabilityProbabilityFormulasAllowed(true);
            multiObjective.setRewardOperatorsAllowed(true);
            multiObjective.setReachabilityRewardFormulasAllowed(true);
            multiObjective.setCumulativeRewardFormulasAllowed(true);
            multiObjective.setBoundedUntilFormulasAllowed(true);
            multiObjective.setStepBoundedUntilFormulasAllowed(true);
 //           multiObjective.setTimeBoundedUntilFormulasAllowed(true); //probably better to activate this only when an MA is given...

            return multiObjective;
        }
        
        FragmentSpecification::FragmentSpecification() {
            probabilityOperator = false;
            rewardOperator = false;
            expectedTimeOperator = false;
            longRunAverageOperator = false;
            
            multiObjectiveFormula = false;
            
            globallyFormula = false;
            reachabilityProbabilityFormula = false;
            nextFormula = false;
            untilFormula = false;
            boundedUntilFormula = false;
            
            atomicExpressionFormula = false;
            atomicLabelFormula = false;
            booleanLiteralFormula = false;
            unaryBooleanStateFormula = false;
            binaryBooleanStateFormula = false;
            
            cumulativeRewardFormula = false;
            instantaneousRewardFormula = false;
            reachabilityRewardFormula = false;
            longRunAverageRewardFormula = false;
            
            conditionalProbabilityFormula = false;
            conditionalRewardFormula = false;
            
            reachabilityTimeFormula = false;
            
            nestedOperators = true;
            nestedPathFormulas = false;
            nestedMultiObjectiveFormulas = false;
            nestedOperatorsInsideMultiObjectiveFormulas = false;
            onlyEventuallyFormuluasInConditionalFormulas = true;
            stepBoundedUntilFormulas = false;
            timeBoundedUntilFormulas = false;
            varianceAsMeasureType = false;
            
            qualitativeOperatorResults = true;
            quantitativeOperatorResults = true;
            
            operatorAtTopLevelRequired = false;
        }
        
        FragmentSpecification FragmentSpecification::copy() const {
            return FragmentSpecification(*this);
        }
        
        bool FragmentSpecification::areProbabilityOperatorsAllowed() const {
            return probabilityOperator;
        }
        
        FragmentSpecification& FragmentSpecification::setProbabilityOperatorsAllowed(bool newValue) {
            this->probabilityOperator = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areRewardOperatorsAllowed() const {
            return rewardOperator;
        }
        
        FragmentSpecification& FragmentSpecification::setRewardOperatorsAllowed(bool newValue) {
            this->rewardOperator = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areTimeOperatorsAllowed() const {
            return expectedTimeOperator;
        }
        
        FragmentSpecification& FragmentSpecification::setTimeOperatorsAllowed(bool newValue) {
            this->expectedTimeOperator = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areLongRunAverageOperatorsAllowed() const {
            return longRunAverageOperator;
        }
        
        FragmentSpecification& FragmentSpecification::setLongRunAverageOperatorsAllowed(bool newValue) {
            this->longRunAverageOperator = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areMultiObjectiveFormulasAllowed() const {
            return multiObjectiveFormula;
        }
        
        FragmentSpecification& FragmentSpecification::setMultiObjectiveFormulasAllowed( bool newValue) {
            this->multiObjectiveFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areGloballyFormulasAllowed() const {
            return globallyFormula;
        }
        
        FragmentSpecification& FragmentSpecification::setGloballyFormulasAllowed(bool newValue) {
            this->globallyFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areReachabilityProbabilityFormulasAllowed() const {
            return reachabilityProbabilityFormula;
        }
        
        FragmentSpecification& FragmentSpecification::setReachabilityProbabilityFormulasAllowed(bool newValue) {
            this->reachabilityProbabilityFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areNextFormulasAllowed() const {
            return nextFormula;
        }
        
        FragmentSpecification& FragmentSpecification::setNextFormulasAllowed(bool newValue) {
            this->nextFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areUntilFormulasAllowed() const {
            return untilFormula;
        }
        
        FragmentSpecification& FragmentSpecification::setUntilFormulasAllowed(bool newValue) {
            this->untilFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areBoundedUntilFormulasAllowed() const {
            return boundedUntilFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setBoundedUntilFormulasAllowed(bool newValue) {
            this->boundedUntilFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areAtomicExpressionFormulasAllowed() const {
            return atomicExpressionFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setAtomicExpressionFormulasAllowed(bool newValue) {
            this->atomicExpressionFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areAtomicLabelFormulasAllowed() const {
            return atomicLabelFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setAtomicLabelFormulasAllowed(bool newValue) {
            this->atomicLabelFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areBooleanLiteralFormulasAllowed() const {
            return booleanLiteralFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setBooleanLiteralFormulasAllowed(bool newValue) {
            this->booleanLiteralFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areUnaryBooleanStateFormulasAllowed() const {
            return unaryBooleanStateFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setUnaryBooleanStateFormulasAllowed(bool newValue) {
            this->unaryBooleanStateFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areBinaryBooleanStateFormulasAllowed() const {
            return binaryBooleanStateFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setBinaryBooleanStateFormulasAllowed(bool newValue) {
            this->binaryBooleanStateFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areCumulativeRewardFormulasAllowed() const {
            return cumulativeRewardFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setCumulativeRewardFormulasAllowed(bool newValue) {
            this->cumulativeRewardFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areInstantaneousRewardFormulasAllowed() const {
            return instantaneousRewardFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setInstantaneousFormulasAllowed(bool newValue) {
            this->instantaneousRewardFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areReachabilityRewardFormulasAllowed() const {
            return reachabilityRewardFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setReachabilityRewardFormulasAllowed(bool newValue) {
            this->reachabilityRewardFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areLongRunAverageRewardFormulasAllowed() const {
            return longRunAverageRewardFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setLongRunAverageRewardFormulasAllowed(bool newValue) {
            this->longRunAverageRewardFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areConditionalProbabilityFormulasAllowed() const {
            return conditionalProbabilityFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setConditionalProbabilityFormulasAllowed(bool newValue) {
            this->conditionalProbabilityFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areConditionalRewardFormulasFormulasAllowed() const {
            return conditionalRewardFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setConditionalRewardFormulasAllowed(bool newValue) {
            this->conditionalRewardFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areReachbilityTimeFormulasAllowed() const {
            return reachabilityTimeFormula;
        }
            
        FragmentSpecification& FragmentSpecification::setReachbilityTimeFormulasAllowed(bool newValue) {
            this->reachabilityTimeFormula = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areNestedOperatorsAllowed() const {
            return this->nestedOperators;
        }
            
        FragmentSpecification& FragmentSpecification::setNestedOperatorsAllowed(bool newValue) {
            this->nestedOperators = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areNestedPathFormulasAllowed() const {
            return this->nestedPathFormulas;
        }
        
        FragmentSpecification& FragmentSpecification::setNestedPathFormulasAllowed(bool newValue) {
            this->nestedPathFormulas = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areNestedMultiObjectiveFormulasAllowed() const {
            return this->nestedMultiObjectiveFormulas;
        }
        
        FragmentSpecification& FragmentSpecification::setNestedMultiObjectiveFormulasAllowed(bool newValue) {
            this->nestedMultiObjectiveFormulas = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areNestedOperatorsInsideMultiObjectiveFormulasAllowed() const {
            return this->nestedOperatorsInsideMultiObjectiveFormulas;
        }
        
        FragmentSpecification& FragmentSpecification::setNestedOperatorsInsideMultiObjectiveFormulasAllowed(bool newValue) {
            this->nestedOperatorsInsideMultiObjectiveFormulas = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areOnlyEventuallyFormuluasInConditionalFormulasAllowed() const {
            return this->onlyEventuallyFormuluasInConditionalFormulas;
        }
        
        FragmentSpecification& FragmentSpecification::setOnlyEventuallyFormuluasInConditionalFormulasAllowed(bool newValue) {
            this->onlyEventuallyFormuluasInConditionalFormulas = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areStepBoundedUntilFormulasAllowed() const {
            return this->stepBoundedUntilFormulas;
        }
        
        FragmentSpecification& FragmentSpecification::setStepBoundedUntilFormulasAllowed(bool newValue) {
            this->stepBoundedUntilFormulas = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areTimeBoundedUntilFormulasAllowed() const {
            return this->timeBoundedUntilFormulas;
        }
        
        FragmentSpecification& FragmentSpecification::setTimeBoundedUntilFormulasAllowed(bool newValue) {
            this->timeBoundedUntilFormulas = newValue;
            return *this;
        }
        
        FragmentSpecification& FragmentSpecification::setOperatorsAllowed(bool newValue) {
            this->setProbabilityOperatorsAllowed(newValue);
            this->setRewardOperatorsAllowed(newValue);
            this->setLongRunAverageOperatorsAllowed(newValue);
            this->setTimeOperatorsAllowed(newValue);
            return *this;
        }
        
        FragmentSpecification& FragmentSpecification::setTimeAllowed(bool newValue) {
            this->setTimeOperatorsAllowed(newValue);
            this->setReachbilityTimeFormulasAllowed(newValue);
            return *this;
        }
        
        FragmentSpecification& FragmentSpecification::setLongRunAverageProbabilitiesAllowed(bool newValue) {
            this->setLongRunAverageOperatorsAllowed(newValue);
            return *this;
        }
        
        bool FragmentSpecification::isVarianceMeasureTypeAllowed() const {
            return varianceAsMeasureType;
        }
        
        FragmentSpecification& FragmentSpecification::setVarianceMeasureTypeAllowed(bool newValue) {
            this->varianceAsMeasureType = newValue;
            return *this;
        }
     
        bool FragmentSpecification::areQuantitativeOperatorResultsAllowed() const {
            return this->quantitativeOperatorResults;
        }
        
        FragmentSpecification& FragmentSpecification::setQuantitativeOperatorResultsAllowed(bool newValue) {
            this->quantitativeOperatorResults = newValue;
            return *this;
        }
        
        bool FragmentSpecification::areQualitativeOperatorResultsAllowed() const {
            return this->qualitativeOperatorResults;
        }
        
        FragmentSpecification& FragmentSpecification::setQualitativeOperatorResultsAllowed(bool newValue) {
            this->qualitativeOperatorResults = newValue;
            return *this;
        }

        bool FragmentSpecification::isOperatorAtTopLevelRequired() const {
            return operatorAtTopLevelRequired;
        }
        
        FragmentSpecification& FragmentSpecification::setOperatorAtTopLevelRequired(bool newValue) {
            operatorAtTopLevelRequired = newValue;
            return *this;
        }
        
    }
}