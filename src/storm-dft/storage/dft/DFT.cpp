#include "DFT.h"

#include <boost/container/flat_set.hpp>
#include <map>

#include "storm/exceptions/NotSupportedException.h"
#include "storm/utility/iota_n.h"
#include "storm/utility/vector.h"

#include "storm-dft/storage/dft/DFTBuilder.h"
#include "storm-dft/storage/dft/DFTIsomorphism.h"


namespace storm {
    namespace storage {

        template<typename ValueType>
        DFT<ValueType>::DFT(DFTElementVector const& elements, DFTElementPointer const& tle) : mElements(elements), mNrOfBEs(0), mNrOfSpares(0), mTopLevelIndex(tle->id()), mMaxSpareChildCount(0) {
            STORM_LOG_ASSERT(elementIndicesCorrect(), "Ids incorrect.");
            size_t nrRepresentatives = 0;
            
            for (auto& elem : mElements) {
                if (isRepresentative(elem->id())) {
                    ++nrRepresentatives;
                }
                if(elem->isBasicElement()) {
                    ++mNrOfBEs;
                }
                else if (elem->isSpareGate()) {
                    ++mNrOfSpares;
                    mMaxSpareChildCount = std::max(mMaxSpareChildCount, std::static_pointer_cast<DFTSpare<ValueType>>(elem)->children().size());
                    for(auto const& spareReprs : std::static_pointer_cast<DFTSpare<ValueType>>(elem)->children()) {
                        std::set<size_t> module = {spareReprs->id()};
                        spareReprs->extendSpareModule(module);
                        std::vector<size_t> sparesAndBes;
                        for(size_t modelem : module) {
                            if(mElements[modelem]->isSpareGate() || mElements[modelem]->isBasicElement()) {
                                sparesAndBes.push_back(modelem);
                                mRepresentants.insert(std::make_pair(modelem, spareReprs->id()));
                            }
                        }
                        mSpareModules.insert(std::make_pair(spareReprs->id(), sparesAndBes));
                    }

                } else if (elem->isDependency()) {
                    mDependencies.push_back(elem->id());
                }
            }
          
            // For the top module, we assume, contrary to [Jun15], that we have all spare gates and basic elements which are not in another module.
            std::set<size_t> topModuleSet;
            // Initialize with all ids.
            for(auto const& elem : mElements) {
                if (elem->isBasicElement() || elem->isSpareGate()) {
                    topModuleSet.insert(elem->id());
                }
            }
            // Erase spare modules
            for(auto const& module : mSpareModules) {
                for(auto const& index : module.second) {
                    topModuleSet.erase(index);
                }
            }
            // Extend top module and insert those elements which are part of the top module and a spare module
            mElements[mTopLevelIndex]->extendSpareModule(topModuleSet);
            mTopModule = std::vector<size_t>(topModuleSet.begin(), topModuleSet.end());
            // Clear all spare modules where at least one element is also in the top module
            if (!mTopModule.empty()) {
                for (auto& module : mSpareModules) {
                    if (std::find(module.second.begin(), module.second.end(), mTopModule.front()) != module.second.end()) {
                        module.second.clear();
                    }
                }
            }

            //Reserve space for failed spares
            ++mMaxSpareChildCount;
            size_t usageInfoBits = storm::utility::math::uint64_log2(mMaxSpareChildCount) + 1;
            mStateVectorSize = nrElements() * 2 + mNrOfSpares * usageInfoBits + nrRepresentatives;
        }

        template<typename ValueType>
        DFTStateGenerationInfo DFT<ValueType>::buildStateGenerationInfo(storm::storage::DFTIndependentSymmetries const& symmetries) const {
            DFTStateGenerationInfo generationInfo(nrElements(), mMaxSpareChildCount);
            
            // Generate Pre and Post info for restrictions
            for(auto const& elem : mElements) {
                if(!elem->isDependency() && !elem->isRestriction()) {
                    generationInfo.setRestrictionPreElements(elem->id(), elem->seqRestrictionPres());
                    generationInfo.setRestrictionPostElements(elem->id(), elem->seqRestrictionPosts());
                }
            }

            size_t stateIndex = 0;
            std::queue<size_t> visitQueue;
            storm::storage::BitVector visited(nrElements(), false);

            if (symmetries.groups.empty()) {
                // Perform DFS for whole tree
                visitQueue.push(mTopLevelIndex);
                stateIndex = performStateGenerationInfoDFS(generationInfo, visitQueue, visited, stateIndex);
            } else {
                // Generate information according to symmetries
                for (size_t symmetryIndex : symmetries.sortedSymmetries) {
                    STORM_LOG_ASSERT(!visited[symmetryIndex], "Element already considered for symmetry.");
                    auto const& symmetryGroup = symmetries.groups.at(symmetryIndex);
                    STORM_LOG_ASSERT(!symmetryGroup.empty(), "No symmetry available.");

                    // Insert all elements of first subtree of each symmetry
                    size_t groupIndex = stateIndex;
                    for (std::vector<size_t> const& symmetryElement : symmetryGroup) {
                        if (visited[symmetryElement[0]]) {
                            groupIndex = std::min(groupIndex, generationInfo.getStateIndex(symmetryElement[0]));
                        } else {
                            stateIndex = generateStateInfo(generationInfo, symmetryElement[0], visited, stateIndex);
                        }
                    }
                    size_t offset = stateIndex - groupIndex;
                    
                    // Mirror symmetries
                    size_t noSymmetricElements = symmetryGroup.front().size();
                    STORM_LOG_ASSERT(noSymmetricElements > 1, "No symmetry available.");

                    for (std::vector<size_t> symmetricElements : symmetryGroup) {
                        STORM_LOG_ASSERT(symmetricElements.size() == noSymmetricElements, "No. of symmetric elements do not coincide.");
                        if (visited[symmetricElements[1]]) {
                            // Elements already mirrored
                            for (size_t index : symmetricElements) {
                                STORM_LOG_ASSERT(visited[index], "Element not mirrored.");
                            }
                            continue;
                        }

                        // Initialize for original element
                        size_t originalElement = symmetricElements[0];
                        size_t index = generationInfo.getStateIndex(originalElement);
                        size_t activationIndex = isRepresentative(originalElement) ? generationInfo.getSpareActivationIndex(originalElement) : 0;
                        size_t usageIndex = mElements[originalElement]->isSpareGate() ? generationInfo.getSpareUsageIndex(originalElement) : 0;
                        
                        // Mirror symmetry for each element
                        for (size_t i = 1; i < symmetricElements.size(); ++i) {
                            size_t symmetricElement = symmetricElements[i];
                            visited.set(symmetricElement);

                            generationInfo.addStateIndex(symmetricElement, index + offset * i);
                            stateIndex += 2;

                            STORM_LOG_ASSERT((activationIndex > 0) == isRepresentative(symmetricElement), "Bits for representative incorrect.");
                            if (activationIndex > 0) {
                                generationInfo.addSpareActivationIndex(symmetricElement, activationIndex + offset * i);
                                ++stateIndex;
                            }

                            STORM_LOG_ASSERT((usageIndex > 0) == mElements[symmetricElement]->isSpareGate(), "Bits for usage incorrect.");
                            if (usageIndex > 0) {
                                generationInfo.addSpareUsageIndex(symmetricElement, usageIndex + offset * i);
                                stateIndex += generationInfo.usageInfoBits();
                            }
                        }
                    }

                    // Store starting indices of symmetry groups
                    std::vector<size_t> symmetryIndices;
                    for (size_t i = 0; i < noSymmetricElements; ++i) {
                        symmetryIndices.push_back(groupIndex + i * offset);
                    }
                    generationInfo.addSymmetry(offset, symmetryIndices);
                }
            }

            // TODO symmetries in dependencies?
            // Consider dependencies
            for (size_t idDependency : getDependencies()) {
                std::shared_ptr<DFTDependency<ValueType> const> dependency = getDependency(idDependency);
                visitQueue.push(dependency->id());
                visitQueue.push(dependency->triggerEvent()->id());
                STORM_LOG_THROW(dependency->dependentEvents().size() == 1, storm::exceptions::NotSupportedException, "Direct state generation does not support n-ary dependencies. Consider rewriting them by setting the binary dependency flag.");
                visitQueue.push(dependency->dependentEvents()[0]->id());
            }
            stateIndex = performStateGenerationInfoDFS(generationInfo, visitQueue, visited, stateIndex);

            // Visit all remaining states
            for (size_t i = 0; i < visited.size(); ++i) {
                if (!visited[i]) {
                    visitQueue.push(i);
                    stateIndex = performStateGenerationInfoDFS(generationInfo, visitQueue, visited, stateIndex);
                }
            }
            
            generationInfo.generateSymmetries(symmetries);

            STORM_LOG_TRACE(generationInfo);
            STORM_LOG_ASSERT(stateIndex == mStateVectorSize, "Id incorrect.");
            STORM_LOG_ASSERT(visited.full(), "Not all elements considered.");
            
            return generationInfo;
        }
        
        template<typename ValueType>
        size_t DFT<ValueType>::generateStateInfo(DFTStateGenerationInfo& generationInfo, size_t id, storm::storage::BitVector& visited, size_t stateIndex) const {
            STORM_LOG_ASSERT(!visited[id], "Element already visited.");
            visited.set(id);
            
            // Reserve bits for element
            generationInfo.addStateIndex(id, stateIndex);
            stateIndex += 2;
            
            if (isRepresentative(id)) {
                generationInfo.addSpareActivationIndex(id, stateIndex);
                ++stateIndex;
            }
            
            if (mElements[id]->isSpareGate()) {
                generationInfo.addSpareUsageIndex(id, stateIndex);
                stateIndex += generationInfo.usageInfoBits();
            }
            
            return stateIndex;
        }
        
        template<typename ValueType>
        size_t DFT<ValueType>::performStateGenerationInfoDFS(DFTStateGenerationInfo& generationInfo, std::queue<size_t>& visitQueue, storm::storage::BitVector& visited, size_t stateIndex) const {
            while (!visitQueue.empty()) {
                size_t id = visitQueue.front();
                visitQueue.pop();
                if (visited[id]) {
                    // Already visited
                    continue;
                }
                stateIndex = generateStateInfo(generationInfo, id, visited, stateIndex);
                
                // Insert children
                if (mElements[id]->isGate()) {
                    for (auto const& child : std::static_pointer_cast<DFTGate<ValueType>>(mElements[id])->children()) {
                        visitQueue.push(child->id());
                    }
                }
            }
            return stateIndex;
        }
        
        template<typename ValueType>
        std::vector<DFT<ValueType>>  DFT<ValueType>::topModularisation() const {
            STORM_LOG_ASSERT(isGate(mTopLevelIndex), "Top level element is no gate.");
            auto const& children = getGate(mTopLevelIndex)->children();
            std::map<size_t, std::vector<size_t>> subdfts;
            for(auto const& child : children) {
                std::vector<size_t> isubdft;
                if(child->nrParents() > 1 || child->hasOutgoingDependencies() || child->hasRestrictions()) {
                    STORM_LOG_TRACE("child " << child->name() << " does not allow modularisation.");
                    return {*this};
                }
                if (isGate(child->id())) {
                    isubdft = getGate(child->id())->independentSubDft(false);
                } else {
                    STORM_LOG_ASSERT(isBasicElement(child->id()), "Child is no BE.");
                    if(getBasicElement(child->id())->hasIngoingDependencies()) {
                        STORM_LOG_TRACE("child " << child->name() << "does not allow modularisation.");
                        return {*this};
                    } else {
                        isubdft = {child->id()};
                    }
                    
                }
                if(isubdft.empty()) {
                    return {*this};
                } else {
                    subdfts[child->id()] = isubdft;
                }
            }
            
            std::vector<DFT<ValueType>> res;
            for(auto const& subdft : subdfts) {
                DFTBuilder<ValueType> builder;
            
                for(size_t id : subdft.second) {
                    builder.copyElement(mElements[id]);
                }
                builder.setTopLevel(mElements[subdft.first]->name());
                res.push_back(builder.build());
            }
            return res;
            
        }
        
        template<typename ValueType>
        uint64_t DFT<ValueType>::maxRank() const {
            uint64_t max = 0;
            for (auto const& e : mElements) {
                if(e->rank() > max) {
                    max = e->rank();
                }
            }
            return max;
        }
        
        template<typename ValueType>
        DFT<ValueType> DFT<ValueType>::optimize() const {
            std::vector<size_t> modIdea = findModularisationRewrite();
            STORM_LOG_DEBUG("Modularisation idea: " << storm::utility::vector::toString(modIdea));
            
            if (modIdea.empty()) {
                // No rewrite needed
                return *this;
            }
            
            std::vector<std::vector<size_t>> rewriteIds;
            rewriteIds.push_back(modIdea);
            
            DFTBuilder<ValueType> builder;
            
            // Accumulate elements which must be rewritten
            std::set<size_t> rewriteSet;
            for (std::vector<size_t> rewrites : rewriteIds) {
                rewriteSet.insert(rewrites.front());
            }
            // Copy all other elements which do not change
            for (auto elem : mElements) {
                if (rewriteSet.count(elem->id()) == 0) {
                    builder.copyElement(elem);
                }
            }
            
            // Add rewritten elements
            for (std::vector<size_t> rewrites : rewriteIds) {
                STORM_LOG_ASSERT(rewrites.size() > 1, "No rewritten elements.");
                STORM_LOG_ASSERT(mElements[rewrites[1]]->hasParents(), "Rewritten elements has no parents.");
                STORM_LOG_ASSERT(mElements[rewrites[1]]->parents().front()->isGate(), "Rewritten element has no parent gate.");
                DFTGatePointer originalParent = std::static_pointer_cast<DFTGate<ValueType>>(mElements[rewrites[1]]->parents().front());
                std::string newParentName = builder.getUniqueName(originalParent->name());
                
                // Accumulate children names
                std::vector<std::string> childrenNames;
                for (size_t i = 1; i < rewrites.size(); ++i) {
                    STORM_LOG_ASSERT(mElements[rewrites[i]]->parents().front()->id() == originalParent->id(), "Children have the same father");
                    childrenNames.push_back(mElements[rewrites[i]]->name());
                }
                
                // Add element inbetween parent and children
                switch (originalParent->type()) {
                    case DFTElementType::AND:
                        builder.addAndElement(newParentName, childrenNames);
                        break;
                    case DFTElementType::OR:
                        builder.addOrElement(newParentName, childrenNames);
                        break;
                    case DFTElementType::BE:
                    case DFTElementType::CONSTF:
                    case DFTElementType::CONSTS:
                    case DFTElementType::VOT:
                    case DFTElementType::PAND:
                    case DFTElementType::SPARE:
                    case DFTElementType::POR:
                    case DFTElementType::PDEP:
                    case DFTElementType::SEQ:
                    case DFTElementType::MUTEX:
                        // Other elements are not supported
                    default:
                        STORM_LOG_ASSERT(false, "Dft type can not be rewritten.");
                        break;
                }
                
                // Add parent with the new child newParent and all its remaining children
                childrenNames.clear();
                childrenNames.push_back(newParentName);
                for (auto const& child : originalParent->children()) {
                    if (std::find(rewrites.begin()+1, rewrites.end(), child->id()) == rewrites.end()) {
                        // Child was not rewritten and must be kept
                        childrenNames.push_back(child->name());
                    }
                }
                builder.copyGate(originalParent, childrenNames);
            }
            
            builder.setTopLevel(mElements[mTopLevelIndex]->name());
            // TODO use reference?
            DFT<ValueType> newDft = builder.build();
            STORM_LOG_TRACE(newDft.getElementsString());
            return newDft.optimize();
        }
        
        template<typename ValueType>
        std::string DFT<ValueType>::getElementsString() const {
            std::stringstream stream;
            for (auto const& elem : mElements) {
                stream << "[" << elem->id() << "]" << elem->toString() << std::endl;
            }
            return stream.str();
        }

        template<typename ValueType>
        std::string DFT<ValueType>::getInfoString() const {
            std::stringstream stream;
            stream << "Top level index: " << mTopLevelIndex << ", Nr BEs" << mNrOfBEs;
            return stream.str();
        }

        template<typename ValueType>
        std::string DFT<ValueType>::getSpareModulesString() const {
            std::stringstream stream;
            stream << "[" << mElements[mTopLevelIndex]->id() << "] {";
            std::vector<size_t>::const_iterator it = mTopModule.begin();
            if (it == mTopModule.end()) {
                stream << "}" << std::endl;
                return stream.str();
            }
            STORM_LOG_ASSERT(it != mTopModule.end(), "Element not found.");
            stream << mElements[(*it)]->name();
            ++it;
            while(it != mTopModule.end()) {
                stream <<  ", " << mElements[(*it)]->name();
                ++it;
            }
            stream << "}" << std::endl;

            for(auto const& spareModule : mSpareModules) {
                stream << "[" << mElements[spareModule.first]->name() << "] = {";
                if (!spareModule.second.empty()) {
                    std::vector<size_t>::const_iterator it = spareModule.second.begin();
                    STORM_LOG_ASSERT(it != spareModule.second.end(), "Element not found.");
                    stream << mElements[(*it)]->name();
                    ++it;
                    while(it != spareModule.second.end()) {
                        stream <<  ", " << mElements[(*it)]->name();
                        ++it;
                    }
                }
                stream << "}" << std::endl;
            }
            return stream.str();
        }

        template<typename ValueType>
        std::string DFT<ValueType>::getElementsWithStateString(DFTStatePointer const& state) const{
            std::stringstream stream;
            for (auto const& elem : mElements) {
                stream << "[" << elem->id() << "]";
                stream << elem->toString();
                if (elem->isDependency()) {
                    stream << "\t** " << storm::storage::toChar(state->getDependencyState(elem->id())) << "[dep]";
                } else {
                    stream << "\t** " << storm::storage::toChar(state->getElementState(elem->id()));
                    if(elem->isSpareGate()) {
                        size_t useId = state->uses(elem->id());
                        if(useId == elem->id() || state->isActive(useId)) {
                            stream << "actively ";
                        }
                        stream << "using " << useId;
                    }
                }
                stream << std::endl;
            }
            return stream.str();
        }
        
        template<typename ValueType>
        std::string DFT<ValueType>::getStateString(DFTStatePointer const& state) const {
            std::stringstream stream;
            stream << "(" << state->getId() << ") ";
            for (auto const& elem : mElements) {
                if (elem->isDependency()) {
                    stream << storm::storage::toChar(state->getDependencyState(elem->id())) << "[dep]";
                } else {
                    stream << storm::storage::toChar(state->getElementState(elem->id()));
                    if(elem->isSpareGate()) {
                        stream << "[";
                        size_t useId = state->uses(elem->id());
                        if(useId == elem->id() || state->isActive(useId)) {
                            stream << "actively ";
                        }
                        stream << "using " << useId << "]";
                    }
                }
            }
            return stream.str();
        }        

        template<typename ValueType>
        std::string DFT<ValueType>::getStateString(storm::storage::BitVector const& status, DFTStateGenerationInfo const& stateGenerationInfo, size_t id) const {
            std::stringstream stream;
            stream << "(" << id << ") ";
            for (auto const& elem : mElements) {
                size_t elemIndex = stateGenerationInfo.getStateIndex(elem->id());
                int elementState = DFTState<ValueType>::getElementStateInt(status, elemIndex);
                if (elem->isDependency()) {
                    stream << storm::storage::toChar(static_cast<DFTDependencyState>(elementState)) << "[dep]";
                } else {
                    stream << storm::storage::toChar(static_cast<DFTElementState>(elementState));
                    if(elem->isSpareGate()) {
                        stream << "[";
                        size_t nrUsedChild = status.getAsInt(stateGenerationInfo.getSpareUsageIndex(elem->id()), stateGenerationInfo.usageInfoBits());
                        size_t useId;
                        if (nrUsedChild == getMaxSpareChildCount()) {
                            useId = elem->id();
                        } else {
                            useId = getChild(elem->id(), nrUsedChild);
                        }
                        if(useId == elem->id() || status[stateGenerationInfo.getSpareActivationIndex(useId)]) {
                            stream << "actively ";
                        }
                        stream << "using " << useId << "]";
                    }
                }
            }
            return stream.str();
        }
        
        template<typename ValueType>
        size_t DFT<ValueType>::getChild(size_t spareId, size_t nrUsedChild) const {
            STORM_LOG_ASSERT(mElements[spareId]->isSpareGate(), "Element is no spare.");
            return getGate(spareId)->children()[nrUsedChild]->id();
        }
        
        template<typename ValueType>
        size_t DFT<ValueType>::getNrChild(size_t spareId, size_t childId) const {
            STORM_LOG_ASSERT(mElements[spareId]->isSpareGate(), "Element is no spare.");
            DFTElementVector children = getGate(spareId)->children();
            for (size_t nrChild = 0; nrChild < children.size(); ++nrChild) {
                if (children[nrChild]->id() == childId) {
                    return nrChild;
                }
            }
            STORM_LOG_ASSERT(false, "Child not found.");
            return 0;
        }
        
        template <typename ValueType>
        std::vector<size_t> DFT<ValueType>::getIndependentSubDftRoots(size_t index) const {
            auto elem = getElement(index);
            auto ISD = elem->independentSubDft(false);
            return ISD;
        }

        template<typename ValueType>
        std::vector<size_t> DFT<ValueType>::immediateFailureCauses(size_t index) const {
            if (isGate(index)) {
                STORM_LOG_ASSERT(false, "Is gate.");
                return {};
            } else {
                return {index};
            }
        }

        template<typename ValueType>
        bool DFT<ValueType>::canHaveNondeterminism() const {
            return !getDependencies().empty();
        }

        template<typename ValueType>
        DFTColouring<ValueType> DFT<ValueType>::colourDFT() const {
            return DFTColouring<ValueType>(*this);
        }

        template<typename ValueType>
        std::map<size_t, size_t> DFT<ValueType>::findBijection(size_t index1, size_t index2, DFTColouring<ValueType> const& colouring, bool sparesAsLeaves) const {
            STORM_LOG_TRACE("Considering ids " << index1 << ", " << index2 << " for isomorphism.");
            bool sharedSpareMode = false;
            std::map<size_t, size_t> bijection;
            
            if (isBasicElement(index1)) {
                if (!isBasicElement(index2)) {
                    return {};
                }
                if (colouring.hasSameColour(index1, index2)) {
                    bijection[index1] = index2;
                    return bijection;
                } else {
                    return {};
                }
            }
            
            STORM_LOG_ASSERT(isGate(index1), "Element is no gate.");
            STORM_LOG_ASSERT(isGate(index2), "Element is no gate.");
            std::vector<size_t> isubdft1 = getGate(index1)->independentSubDft(false);
            std::vector<size_t> isubdft2 = getGate(index2)->independentSubDft(false);
            if(isubdft1.empty() || isubdft2.empty() || isubdft1.size() != isubdft2.size()) {
                if (isubdft1.empty() && isubdft2.empty() && sparesAsLeaves) {
                    // Check again for shared spares
                    sharedSpareMode = true;
                    isubdft1 = getGate(index1)->independentSubDft(false, true);
                    isubdft2 = getGate(index2)->independentSubDft(false, true);
                    if(isubdft1.empty() || isubdft2.empty() || isubdft1.size() != isubdft2.size()) {
                        return {};
                    }
                } else {
                    return {};
                }
            }
            STORM_LOG_TRACE("Checking subdfts from " << index1 << ", " << index2 << " for isomorphism.");
            auto LHS = colouring.colourSubdft(isubdft1);
            auto RHS = colouring.colourSubdft(isubdft2);
            auto IsoCheck = DFTIsomorphismCheck<ValueType>(LHS, RHS, *this);
            
            while (IsoCheck.findNextIsomorphism()) {
                bijection = IsoCheck.getIsomorphism();
                if (sharedSpareMode) {
                    bool bijectionSpareCompatible = true;
                    for (size_t elementId : isubdft1) {
                        if (getElement(elementId)->isSpareGate()) {
                            std::shared_ptr<DFTSpare<ValueType>> spareLeft = std::static_pointer_cast<DFTSpare<ValueType>>(mElements[elementId]);
                            std::shared_ptr<DFTSpare<ValueType>> spareRight = std::static_pointer_cast<DFTSpare<ValueType>>(mElements[bijection.at(elementId)]);
                            
                            if (spareLeft->nrChildren() != spareRight->nrChildren()) {
                                bijectionSpareCompatible = false;
                                break;
                            }
                            // Check bijection for spare children
                            for (size_t i = 0; i < spareLeft->nrChildren(); ++i) {
                                size_t childLeftId = spareLeft->children().at(i)->id();
                                size_t childRightId = spareRight->children().at(i)->id();

                                STORM_LOG_ASSERT(bijection.count(childLeftId) == 0, "Child already part of bijection.");
                                if (childLeftId == childRightId) {
                                    // Ignore shared child
                                    continue;
                                }
                                
                                // TODO generalize for more than one parent
                                if (spareLeft->children().at(i)->nrParents() != 1 || spareRight->children().at(i)->nrParents() != 1) {
                                    bijectionSpareCompatible = false;
                                    break;
                                }
                                
                                std::map<size_t, size_t> tmpBijection = findBijection(childLeftId, childRightId, colouring, false);
                                if (!tmpBijection.empty()) {
                                    bijection.insert(tmpBijection.begin(), tmpBijection.end());
                                } else {
                                    bijectionSpareCompatible = false;
                                    break;
                                }
                            }
                            if (!bijectionSpareCompatible) {
                                break;
                            }
                        }
                    }
                    if (bijectionSpareCompatible) {
                        return bijection;
                    }
                } else {
                    return bijection;
                }
            } // end while
            return {};
        }

        template<typename ValueType>
        DFTIndependentSymmetries DFT<ValueType>::findSymmetries(DFTColouring<ValueType> const& colouring) const {
            std::vector<size_t> vec;
            vec.reserve(nrElements());
            storm::utility::iota_n(std::back_inserter(vec), nrElements(), 0);
            BijectionCandidates<ValueType> completeCategories = colouring.colourSubdft(vec);
            std::map<size_t, std::vector<std::vector<size_t>>> res;

            // Find symmetries for gates
            for(auto const& colourClass : completeCategories.gateCandidates) {
                findSymmetriesHelper(colourClass.second, colouring, res);
            }

            // Find symmetries for BEs
            for(auto const& colourClass : completeCategories.beCandidates) {
                findSymmetriesHelper(colourClass.second, colouring, res);
            }

            return DFTIndependentSymmetries(res);
        }

        template<typename ValueType>
        void DFT<ValueType>::findSymmetriesHelper(std::vector<size_t> const& candidates, DFTColouring<ValueType> const& colouring, std::map<size_t, std::vector<std::vector<size_t>>>& result) const {
            if(candidates.size() <= 0) {
                return;
            }

            std::set<size_t> foundEqClassFor;
            for(auto it1 = candidates.cbegin(); it1 != candidates.cend(); ++it1) {
                std::vector<std::vector<size_t>> symClass;
                if(foundEqClassFor.count(*it1) > 0) {
                    // This item is already in a class.
                    continue;
                }
                if(!getElement(*it1)->hasOnlyStaticParents()) {
                    continue;
                }

                std::pair<std::vector<size_t>, std::vector<size_t>> influencedElem1Ids = getSortedParentAndOutDepIds(*it1);
                auto it2 = it1;
                for(++it2; it2 != candidates.cend(); ++it2) {
                    if(!getElement(*it2)->hasOnlyStaticParents()) {
                        continue;
                    }
                    std::vector<size_t> sortedParent2Ids = getElement(*it2)->parentIds();
                    std::sort(sortedParent2Ids.begin(), sortedParent2Ids.end());

                    if(influencedElem1Ids == getSortedParentAndOutDepIds(*it2)) {
                        std::map<size_t, size_t> bijection = findBijection(*it1, *it2, colouring, true);
                        if (!bijection.empty()) {
                            STORM_LOG_TRACE("Subdfts are symmetric");
                            foundEqClassFor.insert(*it2);
                            if(symClass.empty()) {
                                for(auto const& i : bijection) {
                                    symClass.push_back(std::vector<size_t>({i.first}));
                                }
                            }
                            auto symClassIt = symClass.begin();
                            for(auto const& i : bijection) {
                                symClassIt->emplace_back(i.second);
                                ++symClassIt;

                            }
                        }
                    }
                }

                if(!symClass.empty()) {
                    result.emplace(*it1, symClass);
                }
            }
        }

        template<typename ValueType>
        std::vector<size_t> DFT<ValueType>::findModularisationRewrite() const {
           for(auto const& e : mElements) {
               if(e->isGate() && (e->type() == DFTElementType::AND || e->type() == DFTElementType::OR) ) {
                   // suitable parent gate! - Lets check the independent submodules of the children
                   auto const& children = std::static_pointer_cast<DFTGate<ValueType>>(e)->children();
                   for(auto const& child : children) {
                       
                       
                       auto ISD = std::static_pointer_cast<DFTGate<ValueType>>(child)->independentSubDft(true);
                       // In the ISD, check for other children:
                       
                       std::vector<size_t> rewrite = {e->id(), child->id()};
                       for(size_t isdElemId : ISD) {
                           if(isdElemId == child->id()) continue;
                           if(std::find_if(children.begin(), children.end(), [&isdElemId](std::shared_ptr<DFTElement<ValueType>> const& e) { return e->id() == isdElemId; } ) != children.end()) {
                               rewrite.push_back(isdElemId);
                           }
                       }
                       if(rewrite.size() > 2 && rewrite.size() < children.size() - 1) {
                           return rewrite;
                       }
                       
                   }    
               }
           } 
           return {};
        }
    

        template<typename ValueType>
        std::pair<std::vector<size_t>, std::vector<size_t>> DFT<ValueType>::getSortedParentAndOutDepIds(size_t index) const {
            std::pair<std::vector<size_t>, std::vector<size_t>> res;
            res.first = getElement(index)->parentIds();
            std::sort(res.first.begin(), res.first.end());
            for(auto const& dep : getElement(index)->outgoingDependencies()) {
                res.second.push_back(dep->id());
            }
            std::sort(res.second.begin(), res.second.end());
            return res;
        }
        
        // Explicitly instantiate the class.
        template class DFT<double>;

#ifdef STORM_HAVE_CARL
        template class DFT<RationalFunction>;
#endif

    }
}
