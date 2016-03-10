#pragma once

#include <cassert>
#include <vector>
#include <unordered_map>
#include <utility>
#include "DFTElementType.h"
#include "DFTElements.h"
#include "DFT.h"

namespace storm {
namespace storage {


    struct GateGroupToHash {
        static constexpr uint_fast64_t fivebitmask = (1 << 6) - 1;
        static constexpr uint_fast64_t eightbitmask = (1 << 8) - 1;

        /**
         * Hash function, which ensures that the colours are sorted according to their rank.
         */
        uint_fast64_t operator()(DFTElementType type, size_t nrChildren, size_t nrParents, size_t nrPDEPs, size_t rank) const {
            // Sets first bit to 1
            uint_fast64_t groupHash = static_cast<uint_fast64_t>(1) << 63;
            //Assumes 5 bits for the rank,
            groupHash |= (static_cast<uint_fast64_t>(rank) & fivebitmask) << (62 - 5);
            // 8 bits for the nrChildren,
            groupHash |= (static_cast<uint_fast64_t>(nrChildren) & eightbitmask) << (62 - 5 - 8);
            // 5 bits for nrParents,
            groupHash |= (static_cast<uint_fast64_t>(nrParents) & fivebitmask) << (62 - 5 - 8 - 5);
            // 5 bits for nrPDEPs,
            groupHash |= (static_cast<uint_fast64_t>(nrPDEPs) & fivebitmask) << (62 - 5 - 8 - 5 - 5);
            // 5 bits for the type
            groupHash |= (static_cast<uint_fast64_t>(type) & fivebitmask) << (62 - 5 - 8 - 5 - 5 - 5);
            return groupHash;
        }

    };
    
    struct RestrictionGroupToHash {
        static constexpr uint_fast64_t fivebitmask = (1 << 6) - 1;
        
        static constexpr uint_fast64_t eightbitmask = (1 << 8) - 1;
        
        uint_fast64_t operator()(DFTElementType type, size_t nrChildren, size_t rank) const {
            uint_fast64_t groupHash = static_cast<uint_fast64_t>(0);
            groupHash |= (static_cast<uint_fast64_t>(rank) & fivebitmask) << (62 - 5);
            groupHash |= (static_cast<uint_fast64_t>(nrChildren) & eightbitmask) << (62 - 5 - 8);
            groupHash |= (static_cast<uint_fast64_t>(type) & fivebitmask) << (62 - 5 - 8 - 5);
            return groupHash;
        }
    };


    template<typename ValueType>
    struct BEColourClass {
        BEColourClass() = default;
        BEColourClass(ValueType a, ValueType p, size_t h) : aRate(a), pRate(p), hash(h) {}

        ValueType aRate;
        ValueType pRate;
        size_t    hash;
    };

    template<typename ValueType>
    bool operator==(BEColourClass<ValueType> const& lhs, BEColourClass<ValueType> const& rhs) {
        return lhs.hash == rhs.hash && lhs.aRate == rhs.aRate && lhs.pRate == rhs.pRate;
    }

    /**
     *
     */
    template<typename ValueType>
    struct BijectionCandidates {
        std::unordered_map<size_t, std::vector<size_t>> gateCandidates;
        std::unordered_map<BEColourClass<ValueType>, std::vector<size_t>> beCandidates;
        std::unordered_map<std::pair<ValueType, ValueType>, std::vector<size_t>> pdepCandidates;
        std::unordered_map<size_t, std::vector<size_t>> restrictionCandidates;
        
        size_t nrGroups() const {
            return gateCandidates.size() + beCandidates.size() + pdepCandidates.size() + restrictionCandidates.size();
        }
        
        size_t size() const {
            return nrGates() + nrBEs() + nrDeps() + nrRestrictions();
        }
        
        size_t nrGates() const {
            size_t res = 0;
            for(auto const& x : gateCandidates) {
                res += x.second.size();
            }
            return res;
        }
        
        size_t nrBEs() const {
            size_t res = 0;
            for(auto const& x : beCandidates) {
                res += x.second.size();
            }
            return res;
        }
        
        size_t nrDeps() const {
            size_t res = 0;
            for(auto const& x : pdepCandidates) {
                res += x.second.size();
            }
            return res;
        }
        
        size_t nrRestrictions() const {
            size_t res = 0;
            for(auto const& x : restrictionCandidates) {
                res += x.second.size();
            }
            return res;
        }
        
        bool hasGate(size_t index) const {
            for(auto const& x : gateCandidates) {
                for( auto const& ind : x.second) {
                    if(index == ind) return true;
                }
            }
            return false;
        }
        
        bool hasBE(size_t index) const {
            for(auto const& x : beCandidates) {
                for(auto const& ind : x.second) {
                    if(index == ind) return true;
                }
            }
            return false;
        }
        
        bool hasDep(size_t index) const {
            for(auto const& x : pdepCandidates) {
                for(auto const& ind : x.second) {
                    if(index == ind) return true;
                }
            }
            return false;
        }
        
        bool hasRestriction(size_t index) const {
            for(auto const& x : restrictionCandidates) {
                for(auto const& ind : x.second) {
                    if(index == ind) return true;
                }
            }
            return false;
        }
        
        
        bool has(size_t index) const {
            return hasGate(index) || hasBE(index) || hasDep(index) || hasRestriction(index);
        }
        
        
        
        size_t trivialGateGroups() const {
            size_t res = 0;
            for(auto const& x : gateCandidates) {
                if(x.second.size() == 1) ++res;
            }
            return res;
        }
        
        size_t trivialBEGroups() const {
            size_t res = 0;
            for(auto const& x : beCandidates) {
                if(x.second.size() == 1) ++res;
            }
            return res;
        }
    };

    template<typename ValueType>
    class DFTColouring {
        DFT<ValueType> const& dft;
        std::unordered_map<size_t, size_t> gateColour;
        std::unordered_map<size_t, BEColourClass<ValueType>> beColour;
        std::unordered_map<size_t, std::pair<ValueType, ValueType>> depColour;
        std::unordered_map<size_t, size_t> restrictionColour;
        GateGroupToHash gateColourizer;
        RestrictionGroupToHash restrColourizer;

    public:
        DFTColouring(DFT<ValueType> const& ft) : dft(ft) {
            for(size_t id = 0; id < dft.nrElements(); ++id ) {
                if(dft.isBasicElement(id)) {
                    colourize(dft.getBasicElement(id));
                } else if(dft.isGate(id)) {
                    colourize(dft.getGate(id));
                } else if(dft.isDependency(id)) {
                    colourize(dft.getDependency(id));
                } else {
                    assert(dft.isRestriction(id));
                    colourize(dft.getRestriction(id));
                }
            }
        }
        
        bool hasSameColour(size_t index1, size_t index2) const {
            return beColour.at(index1) == beColour.at(index2);
        }


        BijectionCandidates<ValueType> colourSubdft(std::vector<size_t> const& subDftIndices) const {
            BijectionCandidates<ValueType> res;
            for (size_t index : subDftIndices) {
                if(dft.isBasicElement(index)) {
                    auto it = res.beCandidates.find(beColour.at(index));
                    if(it != res.beCandidates.end()) {
                        it->second.push_back(index);
                    } else {
                        res.beCandidates[beColour.at(index)] = std::vector<size_t>({index});
                    }
                } else if(dft.isGate(index)) {
                    auto it = res.gateCandidates.find(gateColour.at(index));
                    if(it != res.gateCandidates.end()) {
                        it->second.push_back(index);
                    } else {
                        res.gateCandidates[gateColour.at(index)] = std::vector<size_t>({index});
                    }
                } else if(dft.isDependency(index)) {
                    auto it = res.pdepCandidates.find(depColour.at(index));
                    if(it != res.pdepCandidates.end()) {
                        it->second.push_back(index);
                    } else {
                        res.pdepCandidates[depColour.at(index)] = std::vector<size_t>({index});
                    }
                } else {
                    assert(dft.isRestriction(index));
                    auto it = res.restrictionCandidates.find(restrictionColour.at(index));
                    if(it != res.restrictionCandidates.end()) {
                        it->second.push_back(index);
                    } else {
                        res.restrictionCandidates[restrictionColour.at(index)] = std::vector<size_t>({index});
                    }
                }

            }
            return res;
        }


    protected:
        void colourize(std::shared_ptr<const DFTBE<ValueType>> const& be) {
             beColour[be->id()] = BEColourClass<ValueType>(be->activeFailureRate(), be->passiveFailureRate(), be->nrParents());
        }

        void colourize(std::shared_ptr<const DFTGate<ValueType>> const& gate) {
             STORM_LOG_TRACE("Colour " << gate->id() << ": "  << gate->type() << " " << gate->nrChildren() << " " << gate->rank() << ".");
             gateColour[gate->id()] = gateColourizer(gate->type(), gate->nrChildren(), gate->nrParents(), 0, gate->rank());
        }
        
        void colourize(std::shared_ptr<const DFTDependency<ValueType>> const& dep) {
            depColour[dep->id()] = std::pair<ValueType, ValueType>(dep->probability(), dep->dependentEvent()->activeFailureRate());
        }
        
        void colourize(std::shared_ptr<const DFTRestriction<ValueType>> const& restr) {
            restrictionColour[restr->id()] = restrColourizer(restr->type(), restr->nrChildren(), restr->rank());
        }
    };




    /**
     * Saves isomorphism between subtrees
     */
    template<typename ValueType>
    class DFTIsomorphismCheck {
        /// Coloured nodes as provided by the input: left hand side
        BijectionCandidates<ValueType> const& bleft;
        /// Coloured nodes as provided by the input: right hand side.
        BijectionCandidates<ValueType> const& bright;
        /// Whether the colourings are compatible
        bool candidatesCompatible = true;
        /// Current bijection
        std::map<size_t, size_t> bijection;
        /// Current permutations of right hand side groups which lead to the homomorphism.
        /// Contains only colours with more than one member.
        BijectionCandidates<ValueType> currentPermutations;
        DFT<ValueType> const& dft;

    public:
        DFTIsomorphismCheck(BijectionCandidates<ValueType> const& left, BijectionCandidates<ValueType> const& right, DFT<ValueType> const& dft) : bleft(left), bright(right), dft(dft)
        {
            checkCompatibility();
        }

        /**
         * Checks whether the candidates are compatible, that is, checks the colours and the number of members for each colour.
         * @return True iff compatible, ie if the preliminary check allows for a isomorphism.
         */
        bool compatible() {
            return candidatesCompatible;
        }


        /**
         * Returns the isomorphism
         * Can only be called after the findIsomorphism procedure returned that an isomorphism has found.
         * @see findIsomorphism
         */
        std::map<size_t, size_t> const& getIsomorphism() const {
            return bijection;
        }

        /**
         * Check whether another isomorphism exists.
         *
         * @return true iff another isomorphism exists.
         */
        bool findNextIsomorphism() {
            if(!candidatesCompatible){
                return false;
            }
            if (bijection.empty()) {
                constructInitialBijection();
            } else {
                if (!findNextBijection()) {
                    return false;
                }
            }
            while(!check()) {
                // continue our search
                if(!findNextBijection()) {
                    // No further bijections to check, no is
                    return false;
                }
            }
            return true;
        }

    protected:
        /**
         * Construct the initial bijection.
         */
        void constructInitialBijection() {
            assert(candidatesCompatible);
            // We first construct the currentPermutations, which helps to determine the current state of the check.
            initializePermutationsAndTreatTrivialGroups(bleft.beCandidates, bright.beCandidates, currentPermutations.beCandidates);
            initializePermutationsAndTreatTrivialGroups(bleft.gateCandidates, bright.gateCandidates, currentPermutations.gateCandidates);
            initializePermutationsAndTreatTrivialGroups(bleft.pdepCandidates, bright.pdepCandidates, currentPermutations.pdepCandidates);
            initializePermutationsAndTreatTrivialGroups(bleft.restrictionCandidates, bright.restrictionCandidates, currentPermutations.restrictionCandidates);
            STORM_LOG_TRACE(bijection.size() << " vs. " << bleft.size() << " vs. " << bright.size());
            assert(bijection.size() == bleft.size());
            
        }

        /**
         * Construct the next bijection
         * @return true if a next bijection exists.
         */
        bool findNextBijection() {
            assert(candidatesCompatible);
            bool foundNext = false;
            if(!currentPermutations.beCandidates.empty()) {
                auto it = currentPermutations.beCandidates.begin();
                while(!foundNext && it != currentPermutations.beCandidates.end()) {
                    foundNext = std::next_permutation(it->second.begin(), it->second.end());
                    ++it;
                }
            }
            if(!foundNext && !currentPermutations.gateCandidates.empty()) {
                auto it = currentPermutations.gateCandidates.begin();
                while(!foundNext && it != currentPermutations.gateCandidates.end()) {
                    foundNext = std::next_permutation(it->second.begin(), it->second.end());
                    ++it;
                }
            }

            if(!foundNext && !currentPermutations.pdepCandidates.empty()) {
                auto it = currentPermutations.pdepCandidates.begin();
                while(!foundNext && it != currentPermutations.pdepCandidates.end()) {
                    foundNext = std::next_permutation(it->second.begin(), it->second.end());
                    ++it;
                }
            }
            
            if(!foundNext && !currentPermutations.restrictionCandidates.empty()) {
                auto it = currentPermutations.restrictionCandidates.begin();
                while(!foundNext && it != currentPermutations.restrictionCandidates.end()) {
                    foundNext = std::next_permutation(it->second.begin(), it->second.end());
                    ++it;
                }
            }
            

            if(foundNext) {
                for(auto const& colour : bleft.beCandidates) {
                    if (colour.second.size() > 1) {
                        assert(currentPermutations.beCandidates.find(colour.first) != currentPermutations.beCandidates.end());
                        zipVectorsIntoMap(colour.second, currentPermutations.beCandidates.find(colour.first)->second, bijection);
                    }
                }

                for(auto const& colour : bleft.gateCandidates) {
                    if (colour.second.size() > 1) {
                        assert(currentPermutations.gateCandidates.find(colour.first) != currentPermutations.gateCandidates.end());
                        zipVectorsIntoMap(colour.second, currentPermutations.gateCandidates.find(colour.first)->second, bijection);
                    }
                }

                for(auto const& colour : bleft.pdepCandidates) {
                    if (colour.second.size() > 1) {
                        assert(currentPermutations.pdepCandidates.find(colour.first) != currentPermutations.pdepCandidates.end());
                        zipVectorsIntoMap(colour.second, currentPermutations.pdepCandidates.find(colour.first)->second, bijection);
                    }
                }
                
                for(auto const& colour : bleft.restrictionCandidates) {
                    if (colour.second.size() > 1) {
                        assert(currentPermutations.restrictionCandidates.find(colour.first) != currentPermutations.restrictionCandidates.end());
                        zipVectorsIntoMap(colour.second, currentPermutations.restrictionCandidates.find(colour.first)->second, bijection);
                    }
                }
            }

            return foundNext;
        }


        /**
         *
         */
        bool check() const {
            assert(bijection.size() == bleft.size());
            // We can skip BEs, as they are identified by they're homomorphic if they are in the same class
            for(auto const& indexpair : bijection) {
                // Check type first. Colouring takes care of a lot, but not necesarily everything (e.g. voting thresholds)
                equalType(*dft.getElement(indexpair.first), *dft.getElement(indexpair.second));
                if(dft.isGate(indexpair.first)) {
                    assert(dft.isGate(indexpair.second));
                    auto const& lGate = dft.getGate(indexpair.first);
                    auto const& rGate = dft.getGate(indexpair.second);
                    if(lGate->isDynamicGate()) {
                        std::vector<size_t> childrenLeftMapped;
                        for(auto const& child : lGate->children() ) {
                            if (bleft.has(child->id())) {
                                childrenLeftMapped.push_back(bijection.at(child->id()));
                            } else {
                                // Indicate shared child which is not part of the symmetry
                                // For dynamic gates the order is important
                                childrenLeftMapped.push_back(-1);
                            }
                        }
                        std::vector<size_t> childrenRight;
                        for(auto const& child : rGate->children() ) {
                            if (bright.has(child->id())) {
                                childrenRight.push_back(child->id());
                            } else {
                                // Indicate shared child which is not part of the symmetry
                                // For dynamic gates the order is important
                                childrenRight.push_back(-1);
                            }
                        }
                        if(childrenLeftMapped != childrenRight) {
                            return false;
                        } 
                    } else {
                        std::set<size_t> childrenLeftMapped;
                        for(auto const& child : lGate->children() ) {
                            if (bleft.has(child->id())) {
                                childrenLeftMapped.insert(bijection.at(child->id()));
                            }
                        }
                        std::set<size_t> childrenRight;
                        for(auto const& child : rGate->children() ) {
                            if (bright.has(child->id())) {
                                childrenRight.insert(child->id());
                            }
                        }
                        if(childrenLeftMapped != childrenRight) {
                            return false;
                        } 
                    }
                
                    
                } else if(dft.isDependency(indexpair.first)) {
                    assert(dft.isDependency(indexpair.second));
                    auto const& lDep = dft.getDependency(indexpair.first);
                    auto const& rDep = dft.getDependency(indexpair.second);
                    if(bijection.at(lDep->triggerEvent()->id()) != rDep->triggerEvent()->id()) {
                        return false;
                    } 
                    if(bijection.at(lDep->dependentEvent()->id()) != rDep->dependentEvent()->id()) {
                        return false;
                    }
                } else if(dft.isRestriction(indexpair.first)) {
                    assert(dft.isRestriction(indexpair.second));
                    auto const& lRestr = dft.getRestriction(indexpair.first);
                    std::vector<size_t> childrenLeftMapped;
                    for(auto const& child : lRestr->children() ) {
                        if (bleft.has(child->id())) {
                            childrenLeftMapped.push_back(bijection.at(child->id()));
                        } else {
                            // Indicate shared child which is not part of the symmetry
                            // For dynamic gates the order is important
                            childrenLeftMapped.push_back(-1);
                        }
                    }
                    auto const& rRestr = dft.getRestriction(indexpair.second);
                    std::vector<size_t> childrenRight;
                    for(auto const& child : rRestr->children() ) {
                        if (bright.has(child->id())) {
                            childrenRight.push_back(child->id());
                        } else {
                            // Indicate shared child which is not part of the symmetry
                            // For dynamic gates the order is important
                            childrenRight.push_back(-1);
                        }
                    }
                    if(childrenLeftMapped != childrenRight) {
                        return false;
                    }
                }
                else {
                    assert(dft.isBasicElement(indexpair.first));
                    assert(dft.isBasicElement(indexpair.second));
                    // No operations required.
                }
            }
            return true;
        }

    private:
        /**
         * Returns true if the colours are compatible.
         */
        bool checkCompatibility() {
            if(bleft.gateCandidates.size() != bright.gateCandidates.size()) {
                candidatesCompatible = false;
                return false;
            }
            if(bleft.beCandidates.size() != bright.beCandidates.size()) {
                candidatesCompatible = false;
                return false;
            }
            if(bleft.beCandidates.size() != bright.beCandidates.size()) {
                candidatesCompatible = false;
                return false;
            }
            if(bleft.restrictionCandidates.size() != bright.restrictionCandidates.size()) {
                candidatesCompatible = false;
                return false;
            }
            
            for (auto const &gc : bleft.gateCandidates) {
                if (bright.gateCandidates.count(gc.first) == 0) {
                    candidatesCompatible = false;
                    return false;
                }
            }
            for(auto const& bc : bleft.beCandidates) {
                if(bright.beCandidates.count(bc.first) == 0) {
                    candidatesCompatible = false;
                    return false;
                }
            }
               
            for(auto const& dc : bleft.pdepCandidates) {
                if(bright.pdepCandidates.count(dc.first) == 0) {
                    candidatesCompatible = false;
                    return false;
                }
            }
            
            for(auto const& dc : bleft.restrictionCandidates) {
                if(bright.restrictionCandidates.count(dc.first) == 0) {
                    candidatesCompatible = false;
                    return false;
                }
            }
            return true;
        }

        /**
         *
         */
        template<typename ColourType>
        void initializePermutationsAndTreatTrivialGroups(std::unordered_map<ColourType, std::vector<size_t>> const& left,
                                                         std::unordered_map<ColourType, std::vector<size_t>> const& right,
                                                         std::unordered_map<ColourType, std::vector<size_t>>& permutations) {
            for(auto const& colour : right) {
                if(colour.second.size()>1) {
                    auto it = permutations.insert(colour);
                    assert(it.second);
                    std::sort(it.first->second.begin(), it.first->second.end());
                    zipVectorsIntoMap(left.at(colour.first), it.first->second, bijection);
                } else {
                    assert(colour.second.size() == 1);
                    assert(bijection.count(left.at(colour.first).front()) == 0);
                    bijection[left.at(colour.first).front()] = colour.second.front();
                }
            }
        }

        /**
         * Local helper function for the creation of bijections, should be hidden from api.
         */
        void zipVectorsIntoMap(std::vector<size_t> const& a, std::vector<size_t> const& b, std::map<size_t, size_t>& map) const {
            // Assert should pass due to compatibility check
            assert(a.size() == b.size());
            auto it = b.cbegin();
            for(size_t lIndex : a) {
                map[lIndex] = *it;
                ++it;
            }
        }


        
    };


} // namespace storm::dft
} // namespace storm

namespace std {
    template<typename ValueType>
    struct hash<storm::storage::BEColourClass<ValueType>> {
        size_t operator()(storm::storage::BEColourClass<ValueType> const& bcc) const {
            std::hash<ValueType> hasher;
            return (hasher(bcc.aRate) ^ hasher(bcc.pRate) << 8) | bcc.hash;
        }
    };

    template<typename ValueType>
    struct hash<std::pair<ValueType, ValueType>> {
        size_t operator()(std::pair<ValueType, ValueType> const& p) const {
            std::hash<ValueType> hasher;
            return hasher(p.first) ^ hasher(p.second);
        }
    };
}