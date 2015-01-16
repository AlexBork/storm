#ifndef STORM_SOLVER_GLPKLPSOLVER_H_
#define STORM_SOLVER_GLPKLPSOLVER_H_

#include "src/solver/LpSolver.h"
#include "src/exceptions/NotImplementedException.h"

// To detect whether the usage of glpk is possible, this include is neccessary.
#include "storm-config.h"

#ifdef STORM_HAVE_GLPK
#include <glpk.h>
#endif

namespace storm {
    namespace solver {
#ifdef STORM_HAVE_GLPK
        /*!
         * A class that implements the LpSolver interface using glpk as the background solver.
         */
        class GlpkLpSolver : public LpSolver {
        public:
            /*!
             * Constructs a solver with the given name and model sense.
             *
             * @param name The name of the LP problem.
             * @param modelSense A value indicating whether the value of the objective function is to be minimized or
             * maximized.
             */
            GlpkLpSolver(std::string const& name, ModelSense const& modelSense);
            
            /*!
             * Constructs a solver with the given name. By default the objective function is assumed to be minimized,
             * but this may be altered later using a call to setModelSense.
             *
             * @param name The name of the LP problem.
             */
            GlpkLpSolver(std::string const& name);
            
            /*!
             * Constructs a solver without a name and the given model sense.
             *
             * @param modelSense A value indicating whether the value of the objective function is to be minimized or
             * maximized.
             */
            GlpkLpSolver(ModelSense const& modelSense);

            /*!
             * Constructs a solver without a name. By default the objective function is assumed to be minimized,
             * but this may be altered later using a call to setModelSense.
             */
            GlpkLpSolver();
            
            /*!
             * Destructs a solver by freeing the pointers to glpk's structures.
             */
            virtual ~GlpkLpSolver();
            
            // Methods to add continuous variables.
            virtual storm::expressions::Variable addBoundedContinuousVariable(std::string const& name, double lowerBound, double upperBound, double objectiveFunctionCoefficient = 0) override;
            virtual storm::expressions::Variable addLowerBoundedContinuousVariable(std::string const& name, double lowerBound, double objectiveFunctionCoefficient = 0) override;
            virtual storm::expressions::Variable addUpperBoundedContinuousVariable(std::string const& name, double upperBound, double objectiveFunctionCoefficient = 0) override;
            virtual storm::expressions::Variable addUnboundedContinuousVariable(std::string const& name, double objectiveFunctionCoefficient = 0) override;
            
            // Methods to add integer variables.
            virtual storm::expressions::Variable addBoundedIntegerVariable(std::string const& name, double lowerBound, double upperBound, double objectiveFunctionCoefficient = 0) override;
            virtual storm::expressions::Variable addLowerBoundedIntegerVariable(std::string const& name, double lowerBound, double objectiveFunctionCoefficient = 0) override;
            virtual storm::expressions::Variable addUpperBoundedIntegerVariable(std::string const& name, double upperBound, double objectiveFunctionCoefficient = 0) override;
            virtual storm::expressions::Variable addUnboundedIntegerVariable(std::string const& name, double objectiveFunctionCoefficient = 0) override;

            // Methods to add binary variables.
            virtual storm::expressions::Variable addBinaryVariable(std::string const& name, double objectiveFunctionCoefficient = 0) override;

            // Methods to incorporate recent changes.
            virtual void update() const override;
            
            // Methods to add constraints
            virtual void addConstraint(std::string const& name, storm::expressions::Expression const& constraint) override;
            
            // Methods to optimize and retrieve optimality status.
            virtual void optimize() const override;
            virtual bool isInfeasible() const override;
            virtual bool isUnbounded() const override;
            virtual bool isOptimal() const override;

            // Methods to retrieve values of variables and the objective function in the optimal solutions.
            virtual double getContinuousValue(storm::expressions::Variable const& name) const override;
            virtual int_fast64_t getIntegerValue(storm::expressions::Variable const& name) const override;
            virtual bool getBinaryValue(storm::expressions::Variable const& name) const override;
            virtual double getObjectiveValue() const override;

            // Methods to print the LP problem to a file.
            virtual void writeModelToFile(std::string const& filename) const override;
            
        private:
            /*!
             * Adds a variable with the given name, type, lower and upper bound and objective function coefficient.
             *
             * @param variable The variable to add.
             * @param variableType The type of the variable in terms of glpk's constants.
             * @param boundType A glpk flag indicating which bounds apply to the variable.
             * @param lowerBound The lower bound of the range of the variable.
             * @param upperBound The upper bound of the range of the variable.
             * @param objectiveFunctionCoefficient The coefficient of the variable in the objective function.
             */
            void addVariable(storm::expressions::Variable const& variable, int variableType, int boundType, double lowerBound, double upperBound, double objectiveFunctionCoefficient);
            
            // The glpk LP problem.
            glp_prob* lp;
            
            // A mapping from variables to their indices.
            std::map<storm::expressions::Variable, int> variableToIndexMap;
            
            // A counter used for getting the next variable index.
            int nextVariableIndex;
            
            // A counter used for getting the next constraint index.
            int nextConstraintIndex;
            
            // A flag storing whether the model is an LP or an MILP.
            bool modelContainsIntegerVariables;
            
            // Flags that store whether the MILP was found to be infeasible or unbounded.
            mutable bool isInfeasibleFlag;
            mutable bool isUnboundedFlag;
            
            // The arrays that store the coefficient matrix of the problem.
            std::vector<int> rowIndices;
            std::vector<int> columnIndices;
            std::vector<double> coefficientValues;
        };
#else
        // If glpk is not available, we provide a stub implementation that emits an error if any of its methods is called.
        class GlpkLpSolver : public LpSolver {
        public:
            GlpkLpSolver(std::string const& name, ModelSense const& modelSense) {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            GlpkLpSolver(std::string const& name) : LpSolver(MINIMIZE) {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            GlpkLpSolver(ModelSense const& modelSense) : LpSolver(modelSense) {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            GlpkLpSolver() : LpSolver(MINIMIZE) {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual ~GlpkLpSolver() {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addBoundedContinuousVariable(std::string const& name, double lowerBound, double upperBound, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addLowerBoundedContinuousVariable(std::string const& name, double lowerBound, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addUpperBoundedContinuousVariable(std::string const& name, double upperBound, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addUnboundedContinuousVariable(std::string const& name, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addBoundedIntegerVariable(std::string const& name, double lowerBound, double upperBound, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addLowerBoundedIntegerVariable(std::string const& name, double lowerBound, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addUpperBoundedIntegerVariable(std::string const& name, double upperBound, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addUnboundedIntegerVariable(std::string const& name, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual storm::expressions::Variable addBinaryVariable(std::string const& name, double objectiveFunctionCoefficient = 0) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual void update() const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual void addConstraint(std::string const& name, storm::expressions::Expression const& constraint) override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual void optimize() const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual bool isInfeasible() const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual bool isUnbounded() const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual bool isOptimal() const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual double getContinuousValue(storm::expressions::Variable const& variable) const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual int_fast64_t getIntegerValue(storm::expressions::Variable const& variable) const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual bool getBinaryValue(storm::expressions::Variable const& variable) const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual double getObjectiveValue() const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
            
            virtual void writeModelToFile(std::string const& filename) const override {
                throw storm::exceptions::NotImplementedException() << "This version of StoRM was compiled without support for glpk. Yet, a method was called that requires this support. Please choose a version of support with glpk support.";
            }
        };
#endif
    }
}

#endif /* STORM_SOLVER_GLPKLPSOLVER_H_ */
