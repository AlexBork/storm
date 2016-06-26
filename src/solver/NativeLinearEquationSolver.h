#ifndef STORM_SOLVER_NATIVELINEAREQUATIONSOLVER_H_
#define STORM_SOLVER_NATIVELINEAREQUATIONSOLVER_H_

#include <ostream>

#include "LinearEquationSolver.h"

namespace storm {
    namespace solver {
        
        template<typename ValueType>
        class NativeLinearEquationSolverSettings {
        public:
            enum class SolutionMethod {
                Jacobi, GaussSeidel, SOR
            };

            NativeLinearEquationSolverSettings();
            
            void setSolutionMethod(SolutionMethod const& method);
            void setPrecision(ValueType precision);
            void setMaximalNumberOfIterations(uint64_t maximalNumberOfIterations);
            void setRelativeTerminationCriterion(bool value);
            void setOmega(ValueType omega);
            
            SolutionMethod getSolutionMethod() const;
            ValueType getPrecision() const;
            uint64_t getMaximalNumberOfIterations() const;
            uint64_t getRelativeTerminationCriterion() const;
            ValueType getOmega() const;
            
        private:
            SolutionMethod method;
            double precision;
            bool relative;
            uint_fast64_t maximalNumberOfIterations;
            ValueType omega;
        };
        
        /*!
         * A class that uses Storm's native matrix operations to implement the LinearEquationSolver interface.
         */
        template<typename ValueType>
        class NativeLinearEquationSolver : public LinearEquationSolver<ValueType> {
        public:
            NativeLinearEquationSolver(storm::storage::SparseMatrix<ValueType> const& A, NativeLinearEquationSolverSettings<ValueType> const& settings = NativeLinearEquationSolverSettings<ValueType>());
            NativeLinearEquationSolver(storm::storage::SparseMatrix<ValueType>&& A, NativeLinearEquationSolverSettings<ValueType> const& settings = NativeLinearEquationSolverSettings<ValueType>());
            
            virtual void solveEquationSystem(std::vector<ValueType>& x, std::vector<ValueType> const& b, std::vector<ValueType>* multiplyResult = nullptr) const override;
            virtual void performMatrixVectorMultiplication(std::vector<ValueType>& x, std::vector<ValueType> const* b, uint_fast64_t n = 1, std::vector<ValueType>* multiplyResult = nullptr) const override;
            virtual void performMatrixVectorMultiplication(std::vector<ValueType>& x, std::vector<ValueType>& result, std::vector<ValueType> const* b = nullptr) const override;

            NativeLinearEquationSolverSettings<ValueType>& getSettings();
            NativeLinearEquationSolverSettings<ValueType> const& getSettings() const;

        private:
            // If the solver takes posession of the matrix, we store the moved matrix in this member, so it gets deleted
            // when the solver is destructed.
            std::unique_ptr<storm::storage::SparseMatrix<ValueType>> localA;
            
            // A reference to the original sparse matrix given to this solver. If the solver takes posession of the matrix
            // the reference refers to localA.
            storm::storage::SparseMatrix<ValueType> const& A;
                        
            // The settings used by the solver.
            NativeLinearEquationSolverSettings<ValueType> settings;
        };
        
        template<typename ValueType>
        class NativeLinearEquationSolverFactory : public LinearEquationSolverFactory<ValueType> {
        public:
            virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType> const& matrix) const override;
            virtual std::unique_ptr<storm::solver::LinearEquationSolver<ValueType>> create(storm::storage::SparseMatrix<ValueType>&& matrix) const override;
            
            NativeLinearEquationSolverSettings<ValueType>& getSettings();
            NativeLinearEquationSolverSettings<ValueType> const& getSettings() const;
            
            virtual std::unique_ptr<LinearEquationSolverFactory<ValueType>> clone() const override;

        private:
            NativeLinearEquationSolverSettings<ValueType> settings;
        };
    }
}

#endif /* STORM_SOLVER_NATIVELINEAREQUATIONSOLVER_H_ */
