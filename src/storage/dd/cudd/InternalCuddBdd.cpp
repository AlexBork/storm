#include "src/storage/dd/cudd/InternalCuddBdd.h"

#include <boost/functional/hash.hpp>

#include "src/storage/dd/cudd/InternalCuddDdManager.h"
#include "src/storage/dd/Odd.h"

#include "src/storage/BitVector.h"

#include <iostream>

namespace storm {
    namespace dd {
        InternalBdd<DdType::CUDD>::InternalBdd(InternalDdManager<DdType::CUDD> const* ddManager, cudd::BDD cuddBdd) : ddManager(ddManager), cuddBdd(cuddBdd) {
            // Intentionally left empty.
        }
        
        template<typename ValueType>
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::fromVector(InternalDdManager<DdType::CUDD> const* ddManager, std::vector<ValueType> const& values, Odd const& odd, std::vector<uint_fast64_t> const& sortedDdVariableIndices, std::function<bool (ValueType const&)> const& filter) {
            uint_fast64_t offset = 0;
            return InternalBdd<DdType::CUDD>(ddManager, cudd::BDD(ddManager->getCuddManager(), fromVectorRec(ddManager->getCuddManager().getManager(), offset, 0, sortedDdVariableIndices.size(), values, odd, sortedDdVariableIndices, filter)));
        }
        
        bool InternalBdd<DdType::CUDD>::operator==(InternalBdd<DdType::CUDD> const& other) const {
            return this->getCuddBdd() == other.getCuddBdd();
        }
        
        bool InternalBdd<DdType::CUDD>::operator!=(InternalBdd<DdType::CUDD> const& other) const {
            return !(*this == other);
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::relationalProduct(InternalBdd<DdType::CUDD> const& relation, std::vector<InternalBdd<DdType::CUDD>> const& rowVariables, std::vector<InternalBdd<DdType::CUDD>> const& columnVariables) const {
            InternalBdd<DdType::CUDD> cube = ddManager->getBddOne();
            for (auto const& variable : rowVariables) {
                cube &= variable;
            }
            
            InternalBdd<DdType::CUDD> result = this->andExists(relation, cube);
            result = result.swapVariables(rowVariables, columnVariables);
            return result;
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::inverseRelationalProduct(InternalBdd<DdType::CUDD> const& relation, std::vector<InternalBdd<DdType::CUDD>> const& rowVariables, std::vector<InternalBdd<DdType::CUDD>> const& columnVariables) const {
            InternalBdd<DdType::CUDD> cube = ddManager->getBddOne();
            for (auto const& variable : columnVariables) {
                cube &= variable;
            }
            
            InternalBdd<DdType::CUDD> result = this->swapVariables(rowVariables, columnVariables).andExists(relation, cube);
            return result;
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::inverseRelationalProductWithExtendedRelation(InternalBdd<DdType::CUDD> const& relation, std::vector<InternalBdd<DdType::CUDD>> const& rowVariables, std::vector<InternalBdd<DdType::CUDD>> const& columnVariables) const {
            return this->inverseRelationalProduct(relation, rowVariables, columnVariables);
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::ite(InternalBdd<DdType::CUDD> const& thenDd, InternalBdd<DdType::CUDD> const& elseDd) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().Ite(thenDd.getCuddBdd(), elseDd.getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::operator||(InternalBdd<DdType::CUDD> const& other) const {
            InternalBdd<DdType::CUDD> result(*this);
            result |= other;
            return result;
        }
        
        InternalBdd<DdType::CUDD>& InternalBdd<DdType::CUDD>::operator|=(InternalBdd<DdType::CUDD> const& other) {
            this->cuddBdd = this->getCuddBdd() | other.getCuddBdd();
            return *this;
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::operator&&(InternalBdd<DdType::CUDD> const& other) const {
            InternalBdd<DdType::CUDD> result(*this);
            result &= other;
            return result;
        }
        
        InternalBdd<DdType::CUDD>& InternalBdd<DdType::CUDD>::operator&=(InternalBdd<DdType::CUDD> const& other) {
            this->cuddBdd = this->getCuddBdd() & other.getCuddBdd();
            return *this;
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::iff(InternalBdd<DdType::CUDD> const& other) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().Xnor(other.getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::exclusiveOr(InternalBdd<DdType::CUDD> const& other) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().Xor(other.getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::implies(InternalBdd<DdType::CUDD> const& other) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().Ite(other.getCuddBdd(), ddManager->getBddOne().getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::operator!() const {
            InternalBdd<DdType::CUDD> result(*this);
            result.complement();
            return result;
        }
        
        InternalBdd<DdType::CUDD>& InternalBdd<DdType::CUDD>::complement() {
            this->cuddBdd = ~this->getCuddBdd();
            return *this;
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::existsAbstract(InternalBdd<DdType::CUDD> const& cube) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().ExistAbstract(cube.getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::universalAbstract(InternalBdd<DdType::CUDD> const& cube) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().UnivAbstract(cube.getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::andExists(InternalBdd<DdType::CUDD> const& other, InternalBdd<DdType::CUDD> const& cube) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().AndAbstract(other.getCuddBdd(), cube.getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::constrain(InternalBdd<DdType::CUDD> const& constraint) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().Constrain(constraint.getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::restrict(InternalBdd<DdType::CUDD> const& constraint) const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().Restrict(constraint.getCuddBdd()));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::swapVariables(std::vector<InternalBdd<DdType::CUDD>> const& from, std::vector<InternalBdd<DdType::CUDD>> const& to) const {
            std::vector<cudd::BDD> fromBdd;
            std::vector<cudd::BDD> toBdd;
            for (auto it1 = from.begin(), ite1 = from.end(), it2 = to.begin(); it1 != ite1; ++it1, ++it2) {
                fromBdd.push_back(it1->getCuddBdd());
                toBdd.push_back(it2->getCuddBdd());
            }
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().SwapVariables(fromBdd, toBdd));
        }
        
        InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::getSupport() const {
            return InternalBdd<DdType::CUDD>(ddManager, this->getCuddBdd().Support());
        }
        
        uint_fast64_t InternalBdd<DdType::CUDD>::getNonZeroCount(uint_fast64_t numberOfDdVariables) const {
            // If the number of DD variables is zero, CUDD returns a number greater 0 for constant nodes different from
            // zero, which is not the behaviour we expect.
            if (numberOfDdVariables == 0) {
                return 0;
            }
            return static_cast<uint_fast64_t>(this->getCuddBdd().CountMinterm(static_cast<int>(numberOfDdVariables)));
        }
        
        uint_fast64_t InternalBdd<DdType::CUDD>::getLeafCount() const {
            return static_cast<uint_fast64_t>(this->getCuddBdd().CountLeaves());
        }
        
        uint_fast64_t InternalBdd<DdType::CUDD>::getNodeCount() const {
            return static_cast<uint_fast64_t>(this->getCuddBdd().nodeCount());
        }
        
        bool InternalBdd<DdType::CUDD>::isOne() const {
            return this->getCuddBdd().IsOne();
        }
        
        bool InternalBdd<DdType::CUDD>::isZero() const {
            return this->getCuddBdd().IsZero();
        }
        
        uint_fast64_t InternalBdd<DdType::CUDD>::getIndex() const {
            return static_cast<uint_fast64_t>(this->getCuddBdd().NodeReadIndex());
        }
        
        void InternalBdd<DdType::CUDD>::exportToDot(std::string const& filename, std::vector<std::string> const& ddVariableNamesAsStrings) const {
            // Build the name input of the DD.
            std::vector<char*> ddNames;
            std::string ddName("f");
            ddNames.push_back(new char[ddName.size() + 1]);
            std::copy(ddName.c_str(), ddName.c_str() + 2, ddNames.back());
            
            // Now build the variables names.
            std::vector<char*> ddVariableNames;
            for (auto const& element : ddVariableNamesAsStrings) {
                ddVariableNames.push_back(new char[element.size() + 1]);
                std::copy(element.c_str(), element.c_str() + element.size() + 1, ddVariableNames.back());
            }
            
            // Open the file, dump the DD and close it again.
            std::vector<cudd::BDD> cuddBddVector = { this->getCuddBdd() };
            FILE* filePointer = fopen(filename.c_str() , "w");
            ddManager->getCuddManager().DumpDot(cuddBddVector, &ddVariableNames[0], &ddNames[0], filePointer);
            fclose(filePointer);
            
            // Finally, delete the names.
            for (char* element : ddNames) {
                delete element;
            }
            for (char* element : ddVariableNames) {
                delete element;
            }
        }
        
        cudd::BDD InternalBdd<DdType::CUDD>::getCuddBdd() const {
            return this->cuddBdd;
        }
        
        DdNode* InternalBdd<DdType::CUDD>::getCuddDdNode() const {
            return this->getCuddBdd().getNode();
        }
        
        template<typename ValueType>
        InternalAdd<DdType::CUDD, ValueType> InternalBdd<DdType::CUDD>::toAdd() const {
            return InternalAdd<DdType::CUDD, ValueType>(ddManager, this->getCuddBdd().Add());
        }
        
        template<typename ValueType>
        DdNode* InternalBdd<DdType::CUDD>::fromVectorRec(::DdManager* manager, uint_fast64_t& currentOffset, uint_fast64_t currentLevel, uint_fast64_t maxLevel, std::vector<ValueType> const& values, Odd const& odd, std::vector<uint_fast64_t> const& ddVariableIndices, std::function<bool (ValueType const&)> const& filter) {
            if (currentLevel == maxLevel) {
                // If we are in a terminal node of the ODD, we need to check whether the then-offset of the ODD is one
                // (meaning the encoding is a valid one) or zero (meaning the encoding is not valid). Consequently, we
                // need to copy the next value of the vector iff the then-offset is greater than zero.
                if (odd.getThenOffset() > 0) {
                    if (filter(values[currentOffset++])) {
                        return Cudd_ReadOne(manager);
                    } else {
                        return Cudd_ReadLogicZero(manager);
                    }
                } else {
                    return Cudd_ReadZero(manager);
                }
            } else {
                // If the total offset is zero, we can just return the constant zero DD.
                if (odd.getThenOffset() + odd.getElseOffset() == 0) {
                    return Cudd_ReadZero(manager);
                }
                
                // Determine the new else-successor.
                DdNode* elseSuccessor = nullptr;
                if (odd.getElseOffset() > 0) {
                    elseSuccessor = fromVectorRec(manager, currentOffset, currentLevel + 1, maxLevel, values, odd.getElseSuccessor(), ddVariableIndices, filter);
                } else {
                    elseSuccessor = Cudd_ReadLogicZero(manager);
                }
                Cudd_Ref(elseSuccessor);
                
                // Determine the new then-successor.
                DdNode* thenSuccessor = nullptr;
                if (odd.getThenOffset() > 0) {
                    thenSuccessor = fromVectorRec(manager, currentOffset, currentLevel + 1, maxLevel, values, odd.getThenSuccessor(), ddVariableIndices, filter);
                } else {
                    thenSuccessor = Cudd_ReadLogicZero(manager);
                }
                Cudd_Ref(thenSuccessor);
                
                // Create a node representing ITE(currentVar, thenSuccessor, elseSuccessor);
                DdNode* currentVar = Cudd_bddIthVar(manager, static_cast<int>(ddVariableIndices[currentLevel]));
                Cudd_Ref(currentVar);
                DdNode* result = Cudd_bddIte(manager, currentVar, thenSuccessor, elseSuccessor);
                Cudd_Ref(result);
                
                // Dispose of the intermediate results
                Cudd_RecursiveDeref(manager, currentVar);
                Cudd_RecursiveDeref(manager, thenSuccessor);
                Cudd_RecursiveDeref(manager, elseSuccessor);
                
                // Before returning, we remove the protection imposed by the previous call to Cudd_Ref.
                Cudd_Deref(result);
                
                return result;
            }
        }
        
        storm::storage::BitVector InternalBdd<DdType::CUDD>::toVector(storm::dd::Odd const& rowOdd, std::vector<uint_fast64_t> const& ddVariableIndices) const {
            storm::storage::BitVector result(rowOdd.getTotalOffset());
            this->toVectorRec(Cudd_Regular(this->getCuddDdNode()), ddManager->getCuddManager(), result, rowOdd, Cudd_IsComplement(this->getCuddDdNode()), 0, ddVariableIndices.size(), 0, ddVariableIndices);
            return result;
        }
        
        void InternalBdd<DdType::CUDD>::toVectorRec(DdNode const* dd, cudd::Cudd const& manager, storm::storage::BitVector& result, Odd const& rowOdd, bool complement, uint_fast64_t currentRowLevel, uint_fast64_t maxLevel, uint_fast64_t currentRowOffset, std::vector<uint_fast64_t> const& ddRowVariableIndices) const {
            // If there are no more values to select, we can directly return.
            if (dd == Cudd_ReadLogicZero(manager.getManager()) && !complement) {
                return;
            } else if (dd == Cudd_ReadOne(manager.getManager()) && complement) {
                return;
            }
            
            // If we are at the maximal level, the value to be set is stored as a constant in the DD.
            if (currentRowLevel == maxLevel) {
                result.set(currentRowOffset, true);
            } else if (ddRowVariableIndices[currentRowLevel] < dd->index) {
                toVectorRec(dd, manager, result, rowOdd.getElseSuccessor(), complement, currentRowLevel + 1, maxLevel, currentRowOffset, ddRowVariableIndices);
                toVectorRec(dd, manager, result, rowOdd.getThenSuccessor(), complement, currentRowLevel + 1, maxLevel, currentRowOffset + rowOdd.getElseOffset(), ddRowVariableIndices);
            } else {
                // Otherwise, we compute the ODDs for both the then- and else successors.
                DdNode* elseDdNode = Cudd_E(dd);
                DdNode* thenDdNode = Cudd_T(dd);
                
                // Determine whether we have to evaluate the successors as if they were complemented.
                bool elseComplemented = Cudd_IsComplement(elseDdNode) ^ complement;
                bool thenComplemented = Cudd_IsComplement(thenDdNode) ^ complement;
                
                toVectorRec(Cudd_Regular(elseDdNode), manager, result, rowOdd.getElseSuccessor(), elseComplemented, currentRowLevel + 1, maxLevel, currentRowOffset, ddRowVariableIndices);
                toVectorRec(Cudd_Regular(thenDdNode), manager, result, rowOdd.getThenSuccessor(), thenComplemented, currentRowLevel + 1, maxLevel, currentRowOffset + rowOdd.getElseOffset(), ddRowVariableIndices);
            }
        }
        
        Odd InternalBdd<DdType::CUDD>::createOdd(std::vector<uint_fast64_t> const& ddVariableIndices) const {
            // Prepare a unique table for each level that keeps the constructed ODD nodes unique.
            std::vector<std::unordered_map<std::pair<DdNode*, bool>, std::shared_ptr<Odd>, HashFunctor>> uniqueTableForLevels(ddVariableIndices.size() + 1);
            
            // Now construct the ODD structure from the BDD.
            std::shared_ptr<Odd> rootOdd = createOddRec(Cudd_Regular(this->getCuddDdNode()), ddManager->getCuddManager(), 0, Cudd_IsComplement(this->getCuddDdNode()), ddVariableIndices.size(), ddVariableIndices, uniqueTableForLevels);
            
            // Return a copy of the root node to remove the shared_ptr encapsulation.
            return Odd(*rootOdd);
        }
        
        std::size_t InternalBdd<DdType::CUDD>::HashFunctor::operator()(std::pair<DdNode*, bool> const& key) const {
            std::size_t result = 0;
            boost::hash_combine(result, key.first);
            boost::hash_combine(result, key.second);
            return result;
        }
        
        std::shared_ptr<Odd> InternalBdd<DdType::CUDD>::createOddRec(DdNode* dd, cudd::Cudd const& manager, uint_fast64_t currentLevel, bool complement, uint_fast64_t maxLevel, std::vector<uint_fast64_t> const& ddVariableIndices, std::vector<std::unordered_map<std::pair<DdNode*, bool>, std::shared_ptr<Odd>, HashFunctor>>& uniqueTableForLevels) {
            // Check whether the ODD for this node has already been computed (for this level) and if so, return this instead.
            auto const& iterator = uniqueTableForLevels[currentLevel].find(std::make_pair(dd, complement));
            if (iterator != uniqueTableForLevels[currentLevel].end()) {
                return iterator->second;
            } else {
                // Otherwise, we need to recursively compute the ODD.
                
                // If we are already at the maximal level that is to be considered, we can simply create an Odd without
                // successors
                if (currentLevel == maxLevel) {
                    uint_fast64_t elseOffset = 0;
                    uint_fast64_t thenOffset = 0;
                    
                    // If the DD is not the zero leaf, then the then-offset is 1.
                    if (dd != Cudd_ReadZero(manager.getManager())) {
                        thenOffset = 1;
                    }
                    
                    // If we need to complement the 'terminal' node, we need to negate its offset.
                    if (complement) {
                        thenOffset = 1 - thenOffset;
                    }
                    
                    return std::make_shared<Odd>(nullptr, elseOffset, nullptr, thenOffset);
                } else if (ddVariableIndices[currentLevel] < static_cast<uint_fast64_t>(dd->index)) {
                    // If we skipped the level in the DD, we compute the ODD just for the else-successor and use the same
                    // node for the then-successor as well.
                    std::shared_ptr<Odd> elseNode = createOddRec(dd, manager, currentLevel + 1, complement, maxLevel, ddVariableIndices, uniqueTableForLevels);
                    std::shared_ptr<Odd> thenNode = elseNode;
                    uint_fast64_t totalOffset = elseNode->getElseOffset() + elseNode->getThenOffset();
                    return std::make_shared<Odd>(elseNode, totalOffset, thenNode, totalOffset);
                } else {
                    // Otherwise, we compute the ODDs for both the then- and else successors.
                    DdNode* thenDdNode = Cudd_T(dd);
                    DdNode* elseDdNode = Cudd_E(dd);
                    
                    // Determine whether we have to evaluate the successors as if they were complemented.
                    bool elseComplemented = Cudd_IsComplement(elseDdNode) ^ complement;
                    bool thenComplemented = Cudd_IsComplement(thenDdNode) ^ complement;
                    
                    std::shared_ptr<Odd> elseNode = createOddRec(Cudd_Regular(elseDdNode), manager, currentLevel + 1, elseComplemented, maxLevel, ddVariableIndices, uniqueTableForLevels);
                    std::shared_ptr<Odd> thenNode = createOddRec(Cudd_Regular(thenDdNode), manager, currentLevel + 1, thenComplemented, maxLevel, ddVariableIndices, uniqueTableForLevels);
                    
                    return std::make_shared<Odd>(elseNode, elseNode->getElseOffset() + elseNode->getThenOffset(), thenNode, thenNode->getElseOffset() + thenNode->getThenOffset());
                }
            }
        }
        
        template<typename ValueType>
        void InternalBdd<DdType::CUDD>::filterExplicitVector(Odd const& odd, std::vector<uint_fast64_t> const& ddVariableIndices, std::vector<ValueType> const& sourceValues, std::vector<ValueType>& targetValues) const {
            uint_fast64_t currentIndex = 0;
            filterExplicitVectorRec(Cudd_Regular(this->getCuddDdNode()), ddManager->getCuddManager(), 0, Cudd_IsComplement(this->getCuddDdNode()), ddVariableIndices.size(), ddVariableIndices, 0, odd, targetValues, currentIndex, sourceValues);
        }
        
        template<typename ValueType>
        void InternalBdd<DdType::CUDD>::filterExplicitVectorRec(DdNode* dd, cudd::Cudd const& manager, uint_fast64_t currentLevel, bool complement, uint_fast64_t maxLevel, std::vector<uint_fast64_t> const& ddVariableIndices, uint_fast64_t currentOffset, storm::dd::Odd const& odd, std::vector<ValueType>& result, uint_fast64_t& currentIndex, std::vector<ValueType> const& values) {
            // If there are no more values to select, we can directly return.
            if (dd == Cudd_ReadLogicZero(manager.getManager()) && !complement) {
                return;
            } else if (dd == Cudd_ReadOne(manager.getManager()) && complement) {
                return;
            }
            
            if (currentLevel == maxLevel) {
                result[currentIndex++] = values[currentOffset];
            } else if (ddVariableIndices[currentLevel] < dd->index) {
                // If we skipped a level, we need to enumerate the explicit entries for the case in which the bit is set
                // and for the one in which it is not set.
                filterExplicitVectorRec(dd, manager, currentLevel + 1, complement, maxLevel, ddVariableIndices, currentOffset, odd.getElseSuccessor(), result, currentIndex, values);
                filterExplicitVectorRec(dd, manager, currentLevel + 1, complement, maxLevel, ddVariableIndices, currentOffset + odd.getElseOffset(), odd.getThenSuccessor(), result, currentIndex, values);
            } else {
                // Otherwise, we compute the ODDs for both the then- and else successors.
                DdNode* thenDdNode = Cudd_T(dd);
                DdNode* elseDdNode = Cudd_E(dd);
                
                // Determine whether we have to evaluate the successors as if they were complemented.
                bool elseComplemented = Cudd_IsComplement(elseDdNode) ^ complement;
                bool thenComplemented = Cudd_IsComplement(thenDdNode) ^ complement;
                
                filterExplicitVectorRec(Cudd_Regular(elseDdNode), manager, currentLevel + 1, elseComplemented, maxLevel, ddVariableIndices, currentOffset, odd.getElseSuccessor(), result, currentIndex, values);
                filterExplicitVectorRec(Cudd_Regular(thenDdNode), manager, currentLevel + 1, thenComplemented, maxLevel, ddVariableIndices, currentOffset + odd.getElseOffset(), odd.getThenSuccessor(), result, currentIndex, values);
            }
        }
        
        template InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::fromVector(InternalDdManager<DdType::CUDD> const* ddManager, std::vector<double> const& values, Odd const& odd, std::vector<uint_fast64_t> const& sortedDdVariableIndices, std::function<bool (double const&)> const& filter);
        template InternalBdd<DdType::CUDD> InternalBdd<DdType::CUDD>::fromVector(InternalDdManager<DdType::CUDD> const* ddManager, std::vector<uint_fast64_t> const& values, Odd const& odd, std::vector<uint_fast64_t> const& sortedDdVariableIndices, std::function<bool (uint_fast64_t const&)> const& filter);
        
        template InternalAdd<DdType::CUDD, double> InternalBdd<DdType::CUDD>::toAdd() const;
        template InternalAdd<DdType::CUDD, uint_fast64_t> InternalBdd<DdType::CUDD>::toAdd() const;
        
        template void InternalBdd<DdType::CUDD>::filterExplicitVectorRec<double>(DdNode* dd, cudd::Cudd const& manager, uint_fast64_t currentLevel, bool complement, uint_fast64_t maxLevel, std::vector<uint_fast64_t> const& ddVariableIndices, uint_fast64_t currentOffset, storm::dd::Odd const& odd, std::vector<double>& result, uint_fast64_t& currentIndex, std::vector<double> const& values);
        template void InternalBdd<DdType::CUDD>::filterExplicitVectorRec<uint_fast64_t>(DdNode* dd, cudd::Cudd const& manager, uint_fast64_t currentLevel, bool complement, uint_fast64_t maxLevel, std::vector<uint_fast64_t> const& ddVariableIndices, uint_fast64_t currentOffset, storm::dd::Odd const& odd, std::vector<uint_fast64_t>& result, uint_fast64_t& currentIndex, std::vector<uint_fast64_t> const& values);

        template void InternalBdd<DdType::CUDD>::filterExplicitVector(Odd const& odd, std::vector<uint_fast64_t> const& ddVariableIndices, std::vector<double> const& sourceValues, std::vector<double>& targetValues) const;
        template void InternalBdd<DdType::CUDD>::filterExplicitVector(Odd const& odd, std::vector<uint_fast64_t> const& ddVariableIndices, std::vector<uint_fast64_t> const& sourceValues, std::vector<uint_fast64_t>& targetValues) const;
    }
}