#ifndef STORM_UTILITY_SOLVER_H_
#define STORM_UTILITY_SOLVER_H_

#include <set>
#include <vector>
#include <memory>
#include <src/storage/sparse/StateType.h>

#include "src/storage/dd/DdType.h"
#include "src/solver/SolverSelectionOptions.h"

namespace storm {
    namespace solver {
        template<storm::dd::DdType T, typename ValueType>
        class SymbolicGameSolver;
        
        template<storm::dd::DdType T, typename V>
        class SymbolicLinearEquationSolver;
        
        template<storm::dd::DdType T, typename V>
        class SymbolicMinMaxLinearEquationSolver;
        
        template<typename V>
        class GameSolver;
        
        template<typename V>
        class LinearEquationSolver;
        
        template<typename V>
        class MinMaxLinearEquationSolver;
        
        class LpSolver;
        class SmtSolver;
        
        template<typename ValueType> class NativeLinearEquationSolver;
        enum class
                NativeLinearEquationSolverSolutionMethod;
    }

    namespace storage {
        template<typename V> class SparseMatrix;
    }
    
    namespace dd {
        template<storm::dd::DdType Type, typename ValueType>
        class Add;
        
        template<storm::dd::DdType Type>
        class Bdd;
    }
    
    namespace expressions {
        class Variable;
        class ExpressionManager;
    }
    
    namespace utility {
        namespace solver {
            
            template<storm::dd::DdType Type, typename ValueType>
            class SymbolicLinearEquationSolverFactory {
            public:
                virtual std::unique_ptr<storm::solver::SymbolicLinearEquationSolver<Type, ValueType>> create(storm::dd::Add<Type, ValueType> const& A, storm::dd::Bdd<Type> const& allRows, std::set<storm::expressions::Variable> const& rowMetaVariables, std::set<storm::expressions::Variable> const& columnMetaVariables, std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs) const;
            };
            
            template<storm::dd::DdType Type, typename ValueType>
            class SymbolicMinMaxLinearEquationSolverFactory {
                public:
                virtual std::unique_ptr<storm::solver::SymbolicMinMaxLinearEquationSolver<Type, ValueType>> create(storm::dd::Add<Type, ValueType> const& A, storm::dd::Bdd<Type> const& allRows, storm::dd::Bdd<Type> const& illegalMask, std::set<storm::expressions::Variable> const& rowMetaVariables, std::set<storm::expressions::Variable> const& columnMetaVariables, std::set<storm::expressions::Variable> const& choiceVariables, std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs) const;
            };
            
            template<storm::dd::DdType Type, typename ValueType>
            class SymbolicGameSolverFactory {
            public:
                virtual std::unique_ptr<storm::solver::SymbolicGameSolver<Type, ValueType>> create(storm::dd::Add<Type, ValueType> const& A, storm::dd::Bdd<Type> const& allRows, std::set<storm::expressions::Variable> const& rowMetaVariables, std::set<storm::expressions::Variable> const& columnMetaVariables, std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs, std::set<storm::expressions::Variable> const& player1Variables, std::set<storm::expressions::Variable> const& player2Variables) const;
            };
            
            template<typename ValueType>
            class LinearEquationSolverFactory {
            public:
                /*!
                 * Creates a new linear equation solver instance with the given matrix.
                 *
                 * @param matrix The matrix that defines the equation system.
                 * @return A pointer to the newly created solver.
                 */
                virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const;
            };
            
            template<typename ValueType>
            class NativeLinearEquationSolverFactory : public LinearEquationSolverFactory<ValueType> {
            public:
                NativeLinearEquationSolverFactory();
                NativeLinearEquationSolverFactory(typename storm::solver::NativeLinearEquationSolverSolutionMethod method, ValueType omega);
                
                virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const override;
                
            private:
                typename storm::solver::NativeLinearEquationSolverSolutionMethod method;
                ValueType omega;
            };
            
            template<typename ValueType>
            class GmmxxLinearEquationSolverFactory : public LinearEquationSolverFactory<ValueType> {
            public:
                virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const override;
            };
            
            template<typename ValueType>
            class MinMaxLinearEquationSolverFactory {
            public:
                MinMaxLinearEquationSolverFactory(storm::solver::EquationSolverTypeSelection solverType = storm::solver::EquationSolverTypeSelection::FROMSETTINGS);
                /*!
                 * Creates a new min/max linear equation solver instance with the given matrix.
                 */
                virtual std::unique_ptr<storm::solver::MinMaxLinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix, bool trackScheduler = false) const;
                void setSolverType(storm::solver::EquationSolverTypeSelection solverType);
                void setPreferredTechnique(storm::solver::MinMaxTechniqueSelection);
                
            protected:
                /// The type of solver which should be created.
                storm::solver::EquationSolverType solverType;
                /// The preferred technique to be used by the solver.
                /// Notice that we save the selection enum here, which allows different solvers to use different techniques.
                storm::solver::MinMaxTechniqueSelection prefTech;
            };

            template<typename ValueType>
            class GameSolverFactory {
            public:
                /*!
                 * Creates a new game solver instance with the given matrices.
                 */
                virtual std::unique_ptr<storm::solver::GameSolver<ValueType>> create(storm::storage::SparseMatrix<storm::storage::sparse::state_type> const& player1Matrix, storm::storage::SparseMatrix<ValueType> const& player2Matrix) const;
            };
           
            class LpSolverFactory {
            public:
                /*!
                 * Creates a new linear equation solver instance with the given name.
                 *
                 * @param name The name of the LP solver.
                 * @return A pointer to the newly created solver.
                 */
                virtual std::unique_ptr<storm::solver::LpSolver> create(std::string const& name) const;
                virtual std::unique_ptr<storm::solver::LpSolver> create(std::string const& name, storm::solver::LpSolverTypeSelection solvType) const;
            };
            
            class GlpkLpSolverFactory : public LpSolverFactory {
            public:
                virtual std::unique_ptr<storm::solver::LpSolver> create(std::string const& name) const override;
            };
            
            class GurobiLpSolverFactory : public LpSolverFactory {
            public:
                virtual std::unique_ptr<storm::solver::LpSolver> create(std::string const& name) const override;
            };
            
            std::unique_ptr<storm::solver::LpSolver> getLpSolver(std::string const& name, storm::solver::LpSolverTypeSelection solvType = storm::solver::LpSolverTypeSelection::FROMSETTINGS) ;

            class SmtSolverFactory {
            public:
                /*!
                 * Creates a new SMT solver instance.
                 *
                 * @param manager The expression manager responsible for the expressions that will be given to the SMT
                 * solver.
                 * @return A pointer to the newly created solver.
                 */
                virtual std::unique_ptr<storm::solver::SmtSolver> create(storm::expressions::ExpressionManager& manager) const;
            };
            
            class Z3SmtSolverFactory : public SmtSolverFactory {
            public:
                virtual std::unique_ptr<storm::solver::SmtSolver> create(storm::expressions::ExpressionManager& manager) const;
            };

            class MathsatSmtSolverFactory : public SmtSolverFactory {
            public:
                virtual std::unique_ptr<storm::solver::SmtSolver> create(storm::expressions::ExpressionManager& manager) const;
            };
            
            std::unique_ptr<storm::solver::SmtSolver> getSmtSolver(storm::expressions::ExpressionManager& manager);
        }
    }
}

#endif /* STORM_UTILITY_SOLVER_H_ */
