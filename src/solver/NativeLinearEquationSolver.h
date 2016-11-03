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
            
            virtual void setMatrix(storm::storage::SparseMatrix<ValueType> const& A) override;
            virtual void setMatrix(storm::storage::SparseMatrix<ValueType>&& A) override;
            
            virtual bool solveEquations(std::vector<ValueType>& x, std::vector<ValueType> const& b) const override;
            virtual void multiply(std::vector<ValueType>& x, std::vector<ValueType> const* b, std::vector<ValueType>& result) const override;

            NativeLinearEquationSolverSettings<ValueType>& getSettings();
            NativeLinearEquationSolverSettings<ValueType> const& getSettings() const;

            virtual bool allocateAuxMemory(LinearEquationSolverOperation operation) const override;
            virtual bool deallocateAuxMemory(LinearEquationSolverOperation operation) const override;
            virtual bool reallocateAuxMemory(LinearEquationSolverOperation operation) const override;
            virtual bool hasAuxMemory(LinearEquationSolverOperation operation) const override;

            storm::storage::SparseMatrix<ValueType> const& getMatrix() const { return *A; }
        private:
            virtual uint64_t getMatrixRowCount() const override;
            virtual uint64_t getMatrixColumnCount() const override;

            // If the solver takes posession of the matrix, we store the moved matrix in this member, so it gets deleted
            // when the solver is destructed.
            std::unique_ptr<storm::storage::SparseMatrix<ValueType>> localA;
            
            // A pointer to the original sparse matrix given to this solver. If the solver takes posession of the matrix
            // the pointer refers to localA.
            storm::storage::SparseMatrix<ValueType> const* A;
                        
            // The settings used by the solver.
            NativeLinearEquationSolverSettings<ValueType> settings;
            
            // Auxiliary memory for the equation solving methods.
            mutable std::unique_ptr<std::vector<ValueType>> auxiliarySolvingMemory;
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
