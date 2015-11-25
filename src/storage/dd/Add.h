#ifndef STORM_STORAGE_DD_ADD_H_
#define STORM_STORAGE_DD_ADD_H_

#include <map>

#include "src/storage/dd/Dd.h"
#include "src/storage/dd/DdType.h"
#include "src/storage/dd/Odd.h"

#include "src/storage/dd/cudd/InternalCuddAdd.h"
#include "src/storage/dd/sylvan/InternalSylvanAdd.h"

#include "src/storage/dd/cudd/CuddAddIterator.h"
#include "src/storage/dd/sylvan/SylvanAddIterator.h"

namespace storm {
    namespace dd {
        template<DdType LibraryType>
        class Bdd;

        template<DdType LibraryType, typename ValueType>
        class AddIterator;
        
        template<DdType LibraryType, typename ValueType = double>
        class Add : public Dd<LibraryType> {
        public:
            friend class DdManager<LibraryType>;
            friend class Bdd<LibraryType>;
            
            template<DdType LibraryTypePrime, typename ValueTypePrime>
            friend class Add;
            
            // Instantiate all copy/move constructors/assignments with the default implementation.
            Add() = default;
            Add(Add<LibraryType, ValueType> const& other) = default;
            Add& operator=(Add<LibraryType, ValueType> const& other) = default;
            Add(Add<LibraryType, ValueType>&& other) = default;
            Add& operator=(Add<LibraryType, ValueType>&& other) = default;

            /*!
             * Builds an ADD representing the given vector.
             *
             * @param ddManager The manager responsible for the ADD.
             * @param values The vector that is to be represented by the ADD.
             * @param odd The ODD used for the translation.
             * @param metaVariables The meta variables used for the translation.
             * @return The resulting (CUDD) ADD.
             */
            static Add<LibraryType, ValueType> fromVector(std::shared_ptr<DdManager<LibraryType> const> ddManager, std::vector<ValueType> const& values, Odd const& odd, std::set<storm::expressions::Variable> const& metaVariables);
            
            /*!
             * Retrieves whether the two DDs represent the same function.
             *
             * @param other The DD that is to be compared with the current one.
             * @return True if the DDs represent the same function.
             */
            bool operator==(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves whether the two DDs represent different functions.
             *
             * @param other The DD that is to be compared with the current one.
             * @return True if the DDs represent the different functions.
             */
            bool operator!=(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Performs an if-then-else with the given operands, i.e. maps all valuations that are mapped to a non-zero
             * function value to the function values specified by the first DD and all others to the function values
             * specified by the second DD.
             *
             * @param thenDd The ADD specifying the 'then' part.
             * @param elseDd The ADD specifying the 'else' part.
             * @return The ADD corresponding to the if-then-else of the operands.
             */
            Add<LibraryType, ValueType> ite(Add<LibraryType, ValueType> const& thenAdd, Add<LibraryType, ValueType> const& elseAdd) const;
            
            /*!
             * Logically inverts the current ADD. That is, all inputs yielding non-zero values will be mapped to zero in
             * the result and vice versa.
             *
             * @return The resulting ADD.
             */
//            Add<LibraryType, ValueType> operator!() const;
            
            /*!
             * Performs a logical or of the current anBd the given ADD. As a prerequisite, the operand ADDs need to be
             * 0/1 ADDs.
             *
             * @param other The second ADD used for the operation.
             * @return The logical or of the operands.
             */
            Add<LibraryType, ValueType> operator||(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Performs a logical or of the current and the given ADD and assigns it to the current ADD. As a
             * prerequisite, the operand ADDs need to be 0/1 ADDs.
             *
             * @param other The second ADD used for the operation.
             * @return A reference to the current ADD after the operation
             */
            Add<LibraryType, ValueType>& operator|=(Add<LibraryType, ValueType> const& other);
            
            /*!
             * Adds the two ADDs.
             *
             * @param other The ADD to add to the current one.
             * @return The result of the addition.
             */
            Add<LibraryType, ValueType> operator+(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Adds the given ADD to the current one.
             *
             * @param other The ADD to add to the current one.
             * @return A reference to the current ADD after the operation.
             */
            Add<LibraryType, ValueType>& operator+=(Add<LibraryType, ValueType> const& other);
            
            /*!
             * Multiplies the two ADDs.
             *
             * @param other The ADD to multiply with the current one.
             * @return The result of the multiplication.
             */
            Add<LibraryType, ValueType> operator*(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Multiplies the given ADD with the current one and assigns the result to the current ADD.
             *
             * @param other The ADD to multiply with the current one.
             * @return A reference to the current ADD after the operation.
             */
            Add<LibraryType, ValueType>& operator*=(Add<LibraryType, ValueType> const& other);
            
            /*!
             * Subtracts the given ADD from the current one.
             *
             * @param other The ADD to subtract from the current one.
             * @return The result of the subtraction.
             */
            Add<LibraryType, ValueType> operator-(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Subtracts the ADD from the constant zero function.
             *
             * @return The resulting function represented as a ADD.
             */
            Add<LibraryType, ValueType> operator-() const;
            
            /*!
             * Subtracts the given ADD from the current one and assigns the result to the current ADD.
             *
             * @param other The ADD to subtract from the current one.
             * @return A reference to the current ADD after the operation.
             */
            Add<LibraryType, ValueType>& operator-=(Add<LibraryType, ValueType> const& other);
            
            /*!
             * Divides the current ADD by the given one.
             *
             * @param other The ADD by which to divide the current one.
             * @return The result of the division.
             */
            Add<LibraryType, ValueType> operator/(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Divides the current ADD by the given one and assigns the result to the current ADD.
             *
             * @param other The ADD by which to divide the current one.
             * @return A reference to the current ADD after the operation.
             */
            Add<LibraryType, ValueType>& operator/=(Add<LibraryType, ValueType> const& other);
            
            /*!
             * Retrieves the function that maps all evaluations to one that have identical function values.
             *
             * @param other The ADD with which to perform the operation.
             * @return The resulting function represented as an ADD.
             */
            Bdd<LibraryType> equals(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that maps all evaluations to one that have distinct function values.
             *
             * @param other The ADD with which to perform the operation.
             * @return The resulting function represented as an ADD.
             */
            Bdd<LibraryType> notEquals(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that maps all evaluations to one whose function value in the first ADD are less
             * than the one in the given ADD.
             *
             * @param other The ADD with which to perform the operation.
             * @return The resulting function represented as an ADD.
             */
            Bdd<LibraryType> less(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that maps all evaluations to one whose function value in the first ADD are less or
             * equal than the one in the given ADD.
             *
             * @param other The DD with which to perform the operation.
             * @return The resulting function represented as an ADD.
             */
            Bdd<LibraryType> lessOrEqual(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that maps all evaluations to one whose function value in the first ADD are greater
             * than the one in the given ADD.
             *
             * @param other The ADD with which to perform the operation.
             * @return The resulting function represented as an ADD.
             */
            Bdd<LibraryType> greater(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that maps all evaluations to one whose function value in the first ADD are greater
             * or equal than the one in the given ADD.
             *
             * @param other The ADD with which to perform the operation.
             * @return The resulting function represented as an ADD.
             */
            Bdd<LibraryType> greaterOrEqual(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that represents the current ADD to the power of the given ADD.
             *
             * @other The exponent function (given as an ADD).
             * @retur The resulting ADD.
             */
            Add<LibraryType, ValueType> pow(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that represents the current ADD modulo the given ADD.
             *
             * @other The modul function (given as an ADD).
             * @retur The resulting ADD.
             */
            Add<LibraryType, ValueType> mod(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that represents the logarithm of the current ADD to the bases given by the second
             * ADD.
             *
             * @other The base function (given as an ADD).
             * @retur The resulting ADD.
             */
            Add<LibraryType, ValueType> logxy(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that floors all values in the current ADD.
             *
             * @retur The resulting ADD.
             */
            Add<LibraryType, ValueType> floor() const;
            
            /*!
             * Retrieves the function that ceils all values in the current ADD.
             *
             * @retur The resulting ADD.
             */
            Add<LibraryType, ValueType> ceil() const;
            
            /*!
             * Retrieves the function that maps all evaluations to the minimum of the function values of the two ADDs.
             *
             * @param other The ADD with which to perform the operation.
             * @return The resulting function represented as an ADD.
             */
            Add<LibraryType, ValueType> minimum(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Retrieves the function that maps all evaluations to the maximum of the function values of the two ADDs.
             *
             * @param other The ADD with which to perform the operation.
             * @return The resulting function represented as an ADD.
             */
            Add<LibraryType, ValueType> maximum(Add<LibraryType, ValueType> const& other) const;
            
            /*!
             * Sum-abstracts from the given meta variables.
             *
             * @param metaVariables The meta variables from which to abstract.
             */
            Add<LibraryType, ValueType> sumAbstract(std::set<storm::expressions::Variable> const& metaVariables) const;
            
            /*!
             * Min-abstracts from the given meta variables.
             *
             * @param metaVariables The meta variables from which to abstract.
             */
            Add<LibraryType, ValueType> minAbstract(std::set<storm::expressions::Variable> const& metaVariables) const;
            
            /*!
             * Max-abstracts from the given meta variables.
             *
             * @param metaVariables The meta variables from which to abstract.
             */
            Add<LibraryType, ValueType> maxAbstract(std::set<storm::expressions::Variable> const& metaVariables) const;
            
            /*!
             * Checks whether the current and the given ADD represent the same function modulo some given precision.
             *
             * @param other The ADD with which to compare.
             * @param precision An upper bound on the maximal difference between any two function values that is to be
             * tolerated.
             * @param relative If set to true, not the absolute values have to be within the precision, but the relative
             * values.
             */
            bool equalModuloPrecision(Add<LibraryType, ValueType> const& other, double precision, bool relative = true) const;
            
            /*!
             * Swaps the given pairs of meta variables in the ADD. The pairs of meta variables must be guaranteed to have
             * the same number of underlying ADD variables.
             *
             * @param metaVariablePairs A vector of meta variable pairs that are to be swapped for one another.
             * @return The resulting ADD.
             */
            Add<LibraryType, ValueType> swapVariables(std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& metaVariablePairs) const;
            
            /*!
             * Multiplies the current ADD (representing a matrix) with the given matrix by summing over the given meta
             * variables.
             *
             * @param otherMatrix The matrix with which to multiply.
             * @param summationMetaVariables The names of the meta variables over which to sum during the matrix-
             * matrix multiplication.
             * @return An ADD representing the result of the matrix-matrix multiplication.
             */
            Add<LibraryType, ValueType> multiplyMatrix(Add<LibraryType, ValueType> const& otherMatrix, std::set<storm::expressions::Variable> const& summationMetaVariables) const;
            
            /*!
             * Computes a BDD that represents the function in which all assignments with a function value strictly
             * larger than the given value are mapped to one and all others to zero.
             *
             * @param value The value used for the comparison.
             * @return The resulting BDD.
             */
            Bdd<LibraryType> greater(ValueType const& value) const;
            
            /*!
             * Computes a BDD that represents the function in which all assignments with a function value larger or equal
             * to the given value are mapped to one and all others to zero.
             *
             * @param value The value used for the comparison.
             * @return The resulting BDD.
             */
            Bdd<LibraryType> greaterOrEqual(ValueType const& value) const;
            
            /*!
             * Computes a BDD that represents the function in which all assignments with a function value strictly
             * lower than the given value are mapped to one and all others to zero.
             *
             * @param value The value used for the comparison.
             * @return The resulting BDD.
             */
            Bdd<LibraryType> less(ValueType const& value) const;
            
            /*!
             * Computes a BDD that represents the function in which all assignments with a function value less or equal
             * to the given value are mapped to one and all others to zero.
             *
             * @param value The value used for the comparison.
             * @return The resulting BDD.
             */
            Bdd<LibraryType> lessOrEqual(ValueType const& value) const;
            
            /*!
             * Computes a BDD that represents the function in which all assignments with a function value unequal to
             * zero are mapped to one and all others to zero.
             *
             * @return The resulting DD.
             */
            Bdd<LibraryType> notZero() const;
            
            /*!
             * Computes the constraint of the current ADD with the given constraint. That is, the function value of the
             * resulting ADD will be the same as the current ones for all assignments mapping to one in the constraint
             * and may be different otherwise.
             *
             * @param constraint The constraint to use for the operation.
             * @return The resulting ADD.
             */
            Add<LibraryType, ValueType> constrain(Add<LibraryType, ValueType> const& constraint) const;
            
            /*!
             * Computes the restriction of the current ADD with the given constraint. That is, the function value of the
             * resulting DD will be the same as the current ones for all assignments mapping to one in the constraint
             * and may be different otherwise.
             *
             * @param constraint The constraint to use for the operation.
             * @return The resulting ADD.
             */
            Add<LibraryType, ValueType> restrict(Add<LibraryType, ValueType> const& constraint) const;
            
            /*!
             * Retrieves the support of the current ADD.
             *
             * @return The support represented as a BDD.
             */
            Bdd<LibraryType> getSupport() const override;
            
            /*!
             * Retrieves the number of encodings that are mapped to a non-zero value.
             *
             * @return The number of encodings that are mapped to a non-zero value.
             */
            virtual uint_fast64_t getNonZeroCount() const override;
            
            /*!
             * Retrieves the number of leaves of the ADD.
             *
             * @return The number of leaves of the ADD.
             */
            virtual uint_fast64_t getLeafCount() const override;
            
            /*!
             * Retrieves the number of nodes necessary to represent the DD.
             *
             * @return The number of nodes in this DD.
             */
            virtual uint_fast64_t getNodeCount() const override;
            
            /*!
             * Retrieves the lowest function value of any encoding.
             *
             * @return The lowest function value of any encoding.
             */
            ValueType getMin() const;
            
            /*!
             * Retrieves the highest function value of any encoding.
             *
             * @return The highest function value of any encoding.
             */
            ValueType getMax() const;
            
            /*!
             * Sets the function values of all encodings that have the given value of the meta variable to the given
             * target value.
             *
             * @param metaVariable The meta variable that has to be equal to the given value.
             * @param variableValue The value that the meta variable is supposed to have. This must be within the range
             * of the meta variable.
             * @param targetValue The new function value of the modified encodings.
             */
            void setValue(storm::expressions::Variable const& metaVariable, int_fast64_t variableValue, ValueType const& targetValue);
            
            /*!
             * Sets the function values of all encodings that have the given values of the two meta variables to the
             * given target value.
             *
             * @param metaVariable1 The first meta variable that has to be equal to the first given
             * value.
             * @param variableValue1 The value that the first meta variable is supposed to have. This must be within the
             * range of the meta variable.
             * @param metaVariable2 The second meta variable that has to be equal to the second given
             * value.
             * @param variableValue2 The value that the second meta variable is supposed to have. This must be within
             * the range of the meta variable.
             * @param targetValue The new function value of the modified encodings.
             */
            void setValue(storm::expressions::Variable const& metaVariable1, int_fast64_t variableValue1, storm::expressions::Variable const& metaVariable2, int_fast64_t variableValue2, ValueType const& targetValue);
            
            /*!
             * Sets the function values of all encodings that have the given values of the given meta variables to the
             * given target value.
             *
             * @param metaVariableToValueMap A mapping of meta variables to the values they are supposed to have. All
             * values must be within the range of the respective meta variable.
             * @param targetValue The new function value of the modified encodings.
             */
            void setValue(std::map<storm::expressions::Variable, int_fast64_t> const& metaVariableToValueMap = std::map<storm::expressions::Variable, int_fast64_t>(), ValueType const& targetValue = 0);
            
            /*!
             * Retrieves the value of the function when all meta variables are assigned the values of the given mapping.
             * Note that the mapping must specify values for all meta variables contained in the DD.
             *
             * @param metaVariableToValueMap A mapping of meta variables to their values.
             * @return The value of the function evaluated with the given input.
             */
            ValueType getValue(std::map<storm::expressions::Variable, int_fast64_t> const& metaVariableToValueMap = std::map<storm::expressions::Variable, int_fast64_t>()) const;
            
            /*!
             * Retrieves whether this ADD represents the constant one function.
             *
             * @return True if this ADD represents the constant one function.
             */
            bool isOne() const;
            
            /*!
             * Retrieves whether this ADD represents the constant zero function.
             *
             * @return True if this ADD represents the constant zero function.
             */
            bool isZero() const;
            
            /*!
             * Retrieves whether this ADD represents a constant function.
             *
             * @return True if this ADD represents a constants function.
             */
            bool isConstant() const;
            
            /*!
             * Retrieves the index of the topmost variable in the DD.
             *
             * @return The index of the topmost variable in DD.
             */
            virtual uint_fast64_t getIndex() const override;
            
            /*!
             * Converts the ADD to a vector.
             *
             * @return The vector that is represented by this ADD.
             */
            std::vector<ValueType> toVector() const;
            
            /*!
             * Converts the ADD to a vector. The given offset-labeled DD is used to determine the correct row of
             * each entry.
             *
             * @param rowOdd The ODD used for determining the correct row.
             * @return The vector that is represented by this ADD.
             */
            std::vector<ValueType> toVector(storm::dd::Odd const& rowOdd) const;
            
            /*!
             * Converts the ADD to a (sparse) double matrix. All contained non-primed variables are assumed to encode the
             * row, whereas all primed variables are assumed to encode the column.
             *
             * @return The matrix that is represented by this ADD.
             */
            storm::storage::SparseMatrix<ValueType> toMatrix() const;
            
            /*!
             * Converts the ADD to a (sparse) double matrix. All contained non-primed variables are assumed to encode the
             * row, whereas all primed variables are assumed to encode the column. The given offset-labeled DDs are used
             * to determine the correct row and column, respectively, for each entry.
             *
             * @param rowOdd The ODD used for determining the correct row.
             * @param columnOdd The ODD used for determining the correct column.
             * @return The matrix that is represented by this ADD.
             */
            storm::storage::SparseMatrix<ValueType> toMatrix(storm::dd::Odd const& rowOdd, storm::dd::Odd const& columnOdd) const;
            
            /*!
             * Converts the ADD to a (sparse) double matrix. The given offset-labeled DDs are used to determine the
             * correct row and column, respectively, for each entry.
             *
             * @param rowMetaVariables The meta variables that encode the rows of the matrix.
             * @param columnMetaVariables The meta variables that encode the columns of the matrix.
             * @param rowOdd The ODD used for determining the correct row.
             * @param columnOdd The ODD used for determining the correct column.
             * @return The matrix that is represented by this ADD.
             */
            storm::storage::SparseMatrix<ValueType> toMatrix(std::set<storm::expressions::Variable> const& rowMetaVariables, std::set<storm::expressions::Variable> const& columnMetaVariables, storm::dd::Odd const& rowOdd, storm::dd::Odd const& columnOdd) const;
            
            /*!
             * Converts the ADD to a row-grouped (sparse) double matrix. The given offset-labeled DDs are used to
             * determine the correct row and column, respectively, for each entry. Note: this function assumes that
             * the meta variables used to distinguish different row groups are at the very top of the ADD.
             *
             * @param groupMetaVariables The meta variables that are used to distinguish different row groups.
             * @param rowOdd The ODD used for determining the correct row.
             * @param columnOdd The ODD used for determining the correct column.
             * @return The matrix that is represented by this ADD.
             */
            storm::storage::SparseMatrix<ValueType> toMatrix(std::set<storm::expressions::Variable> const& groupMetaVariables, storm::dd::Odd const& rowOdd, storm::dd::Odd const& columnOdd) const;
            
            /*!
             * Converts the ADD to a row-grouped (sparse) double matrix and the given vector to a row-grouped vector.
             * The given offset-labeled DDs are used to determine the correct row and column, respectively, for each
             * entry. Note: this function assumes that the meta variables used to distinguish different row groups are
             * at the very top of the ADD.
             *
             * @param vector The symbolic vector to convert.
             * @param rowGroupSizes A vector specifying the sizes of the row groups.
             * @param groupMetaVariables The meta variables that are used to distinguish different row groups.
             * @param rowOdd The ODD used for determining the correct row.
             * @param columnOdd The ODD used for determining the correct column.
             * @return The matrix that is represented by this ADD.
             */
            std::pair<storm::storage::SparseMatrix<ValueType>, std::vector<ValueType>> toMatrixVector(storm::dd::Add<LibraryType, ValueType> const& vector, std::vector<uint_fast64_t>&& rowGroupSizes, std::set<storm::expressions::Variable> const& groupMetaVariables, storm::dd::Odd const& rowOdd, storm::dd::Odd const& columnOdd) const;
            
            /*!
             * Exports the DD to the given file in the dot format.
             *
             * @param filename The name of the file to which the DD is to be exported.
             */
            void exportToDot(std::string const& filename) const override;
            
            /*!
             * Retrieves an iterator that points to the first meta variable assignment with a non-zero function value.
             *
             * @param enumerateDontCareMetaVariables If set to true, all meta variable assignments are enumerated, even
             * if a meta variable does not at all influence the the function value.
             * @return An iterator that points to the first meta variable assignment with a non-zero function value.
             */
            AddIterator<LibraryType, ValueType> begin(bool enumerateDontCareMetaVariables = true) const;
            
            /*!
             * Retrieves an iterator that points past the end of the container.
             *
             * @param enumerateDontCareMetaVariables If set to true, all meta variable assignments are enumerated, even
             * if a meta variable does not at all influence the the function value.
             * @return An iterator that points past the end of the container.
             */
            AddIterator<LibraryType, ValueType> end(bool enumerateDontCareMetaVariables = true) const;
            
            friend std::ostream & operator<<(std::ostream& out, const Add<LibraryType, ValueType>& add);
            
            /*!
             * Converts the ADD to a BDD by mapping all values unequal to zero to 1. This effectively does the same as
             * a call to notZero().
             *
             * @return The corresponding BDD.
             */
            Bdd<LibraryType> toBdd() const;
            
            /*!
             * Creates an ODD based on the current ADD.
             *
             * @return The corresponding ODD.
             */
            Odd createOdd() const;
            
        private:
            /*!
             * Creates a DD that encapsulates the given CUDD internal ADD.
             *
             * @param ddManager The manager responsible for this DD.
             * @param internalAdd The internal ADD to store.
             * @param containedMetaVariables The meta variables that appear in the DD.
             */
            Add(std::shared_ptr<DdManager<LibraryType> const> ddManager, InternalAdd<LibraryType, ValueType> const& internalAdd, std::set<storm::expressions::Variable> const& containedMetaVariables = std::set<storm::expressions::Variable>());
            
            /*!
             * We provide a conversion operator from the BDD to its internal type to ease calling the internal functions.
             */
            operator InternalAdd<LibraryType, ValueType>() const;
            
            /*!
             * Converts the ADD to a row-grouped (sparse) double matrix. If the optional vector is given, it is also
             * translated to an explicit row-grouped vector with the same row-grouping. The given offset-labeled DDs
             * are used to determine the correct row and column, respectively, for each entry. Note: this function
             * assumes that the meta variables used to distinguish different row groups are at the very top of the ADD.
             *
             * @param rowMetaVariables The meta variables that encode the rows of the matrix.
             * @param columnMetaVariables The meta variables that encode the columns of the matrix.
             * @param groupMetaVariables The meta variables that are used to distinguish different row groups.
             * @param rowOdd The ODD used for determining the correct row.
             * @param columnOdd The ODD used for determining the correct column.
             * @return The matrix that is represented by this ADD and and a vector corresponding to the symbolic vector
             * (if it was given).
             */
            storm::storage::SparseMatrix<ValueType> toMatrix(std::set<storm::expressions::Variable> const& rowMetaVariables, std::set<storm::expressions::Variable> const& columnMetaVariables, std::set<storm::expressions::Variable> const& groupMetaVariables, storm::dd::Odd const& rowOdd, storm::dd::Odd const& columnOdd) const;
            
            /*!
             * Converts the ADD to a row-grouped (sparse) double matrix and the given vector to an equally row-grouped
             * explicit vector. The given offset-labeled DDs are used to determine the correct row and column,
             * respectively, for each entry. Note: this function assumes that the meta variables used to distinguish
             * different row groups are at the very top of the ADD.
             *
             * @param vector The vector that is to be transformed to an equally grouped explicit vector.
             * @param rowGroupSizes A vector specifying the sizes of the row groups.
             * @param rowMetaVariables The meta variables that encode the rows of the matrix.
             * @param columnMetaVariables The meta variables that encode the columns of the matrix.
             * @param groupMetaVariables The meta variables that are used to distinguish different row groups.
             * @param rowOdd The ODD used for determining the correct row.
             * @param columnOdd The ODD used for determining the correct column.
             * @return The matrix that is represented by this ADD and and a vector corresponding to the symbolic vector
             * (if it was given).
             */
            std::pair<storm::storage::SparseMatrix<ValueType>, std::vector<ValueType>> toMatrixVector(storm::dd::Add<LibraryType, ValueType> const& vector, std::vector<uint_fast64_t>&& rowGroupSizes, std::set<storm::expressions::Variable> const& rowMetaVariables, std::set<storm::expressions::Variable> const& columnMetaVariables, std::set<storm::expressions::Variable> const& groupMetaVariables, storm::dd::Odd const& rowOdd, storm::dd::Odd const& columnOdd) const;
                        
            // The internal ADD that depends on the chosen library.
            InternalAdd<LibraryType, ValueType> internalAdd;
        };
    }
}

#endif /* STORM_STORAGE_DD_ADD_H_ */