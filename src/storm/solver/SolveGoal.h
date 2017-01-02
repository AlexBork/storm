#ifndef STORM_SOLVER_SOLVEGOAL_H_
#define	STORM_SOLVER_SOLVEGOAL_H_

#include <memory>

#include "storm/solver/OptimizationDirection.h"
#include "storm/logic/ComparisonType.h"
#include "storm/logic/Bound.h"
#include "storm/storage/BitVector.h"

#include "storm/solver/LinearEquationSolver.h"
#include "storm/solver/MinMaxLinearEquationSolver.h"

namespace storm {
    namespace storage {
        template<typename ValueType> class SparseMatrix;
    }
    
    namespace utility {
        namespace solver {
            template<typename ValueType> class MinMaxLinearEquationSolverFactory;
            template<typename ValueType> class LinearEquationSolverFactory;
        }
    }
    
    namespace solver {
        template<typename ValueType> class MinMaxLinearEquationSolver;
        template<typename ValueType> class LinearEquationSolver;
        
        class SolveGoal {
        public:
            SolveGoal(bool minimize) : optimizationDirection(minimize ? OptimizationDirection::Minimize : OptimizationDirection::Maximize) {
                // Intentionally left empty.
            }
            
            SolveGoal(OptimizationDirection d) : optimizationDirection(d) {
                // Intentionally left empty.
            }
            
            virtual ~SolveGoal() {
                // Intentionally left empty.
            }
           
            bool minimize() const {
                return optimizationDirection == OptimizationDirection::Minimize;
            }
            
            OptimizationDirection direction() const {
                return optimizationDirection;
            }
            
            virtual bool isBounded() const {
                return false;
            }
           
        private:
            OptimizationDirection optimizationDirection;
        };
        
        template<typename ValueType>
        class BoundedGoal : public SolveGoal {
        public:
            BoundedGoal(OptimizationDirection optimizationDirection, storm::logic::ComparisonType comparisonType, ValueType const& threshold, storm::storage::BitVector const& relevantValues) : SolveGoal(optimizationDirection), bound(comparisonType, threshold), relevantValueVector(relevantValues) {
                // Intentionally left empty.
            }
            
            BoundedGoal(OptimizationDirection optimizationDirection, storm::logic::Bound<ValueType> const& bound, storm::storage::BitVector const& relevantValues) : SolveGoal(optimizationDirection), bound(bound), relevantValueVector(relevantValues) {
                // Intentionally left empty.
            }
            
            virtual ~BoundedGoal() {
                // Intentionally left empty.
            }
            
            bool isBounded() const override {
                return true;
            }
            
            bool boundIsALowerBound() const { 
                return (bound.comparisonType == storm::logic::ComparisonType::Greater || bound.comparisonType == storm::logic::ComparisonType::GreaterEqual);
            }
            
            bool boundIsStrict() const {
                return (bound.comparisonType == storm::logic::ComparisonType::Greater || bound.comparisonType == storm::logic::ComparisonType::Less);
            }
            
            ValueType const& thresholdValue() const {
                return bound.threshold;
            }

            bool achieved(std::vector<ValueType> const& result) const{
                for(std::size_t i : relevantValueVector){
                    switch(bound.comparisonType) {
                    case storm::logic::ComparisonType::Greater:
                        if( result[i] <= bound.threshold) return false;
                        break;
                    case storm::logic::ComparisonType::GreaterEqual:
                        if( result[i] < bound.threshold) return false;
                        break;
                    case storm::logic::ComparisonType::Less:
                        if( result[i] >= bound.threshold) return false;
                        break;
                    case storm::logic::ComparisonType::LessEqual:
                        if( result[i] > bound.threshold) return false;
                        break;
                    }
                }
                return true;
            }
       
            storm::storage::BitVector const& relevantValues() const {
                return relevantValueVector;
            }
            
        private:
            Bound<ValueType> bound;
            storm::storage::BitVector relevantValueVector;
        };
        
        template<typename ValueType>
        std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> configureMinMaxLinearEquationSolver(BoundedGoal<ValueType> const& goal, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& factory, storm::storage::SparseMatrix<ValueType> const& matrix);
        
        template<typename ValueType> 
        std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> configureMinMaxLinearEquationSolver(SolveGoal const& goal, storm::solver::MinMaxLinearEquationSolverFactory<ValueType> const& factory, storm::storage::SparseMatrix<ValueType> const& matrix);

        template<typename ValueType>
        std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> configureLinearEquationSolver(BoundedGoal<ValueType> const& goal, storm::solver::LinearEquationSolverFactory<ValueType> const& factory, storm::storage::SparseMatrix<ValueType> const& matrix);
        
        template<typename ValueType>
        std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> configureLinearEquationSolver(SolveGoal const& goal, storm::solver::LinearEquationSolverFactory<ValueType> const& factory, storm::storage::SparseMatrix<ValueType> const& matrix);

    }
}


#endif	/* STORM_SOLVER_SOLVEGOAL_H_ */

