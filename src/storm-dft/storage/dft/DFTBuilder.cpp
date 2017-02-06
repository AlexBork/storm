#include "DFTBuilder.h"

#include <algorithm>

#include "storm/utility/macros.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/exceptions/WrongFormatException.h"

#include "storm-dft/storage/dft/DFT.h"
#include "storm-dft/storage/dft/OrderDFTElementsById.h"


namespace storm {
    namespace storage {
        
        template<typename ValueType>
        std::size_t DFTBuilder<ValueType>::mUniqueOffset = 0;

        template<typename ValueType>
        DFT<ValueType> DFTBuilder<ValueType>::build() {
            for(auto& elem : mChildNames) {
                DFTGatePointer gate = std::static_pointer_cast<DFTGate<ValueType>>(elem.first);
                for(auto const& child : elem.second) {
                    auto itFind = mElements.find(child);
                    if (itFind != mElements.end()) {
                        // Child found
                        DFTElementPointer childElement = itFind->second;
                        STORM_LOG_TRACE("Ignore functional dependency " << child << " in gate " << gate->name());
                        if(!childElement->isDependency()) {
                            gate->pushBackChild(childElement);
                            childElement->addParent(gate);
                        }
                    } else {
                        // Child not found -> find first dependent event to assure that child is dependency
                        // TODO: Not sure whether this is the intended behaviour?
                        auto itFind = mElements.find(child + "_1");
                        STORM_LOG_ASSERT(itFind != mElements.end(), "Child '" << child << "' for gate '" << gate->name() << "' not found.");
                        STORM_LOG_ASSERT(itFind->second->isDependency(), "Child is no dependency.");
                        STORM_LOG_TRACE("Ignore functional dependency " << child << " in gate " << gate->name());
                    }
                }
            }
            
            for(auto& elem : mRestrictionChildNames) {
                for(auto const& childName : elem.second) {
                    auto itFind = mElements.find(childName);
                    STORM_LOG_ASSERT(itFind != mElements.end(), "Child not found.");
                    DFTElementPointer childElement = itFind->second;
                    STORM_LOG_ASSERT(!childElement->isDependency() && !childElement->isRestriction(), "Child has invalid type.");
                    elem.first->pushBackChild(childElement);
                    childElement->addRestriction(elem.first);
                }
            }

            for(auto& elem : mDependencyChildNames) {
                bool first = true;
                std::vector<std::shared_ptr<DFTBE<ValueType>>> dependencies;
                for(auto const& childName : elem.second) {
                    auto itFind = mElements.find(childName);
                    STORM_LOG_ASSERT(itFind != mElements.end(), "Child '" << childName << "' not found");
                    DFTElementPointer childElement = itFind->second;
                    if (!first) {
                        dependencies.push_back(std::static_pointer_cast<DFTBE<ValueType>>(childElement));
                    } else {
                        elem.first->setTriggerElement(std::static_pointer_cast<DFTGate<ValueType>>(childElement));
                        childElement->addOutgoingDependency(elem.first);
                    }
                    first = false;
                }
                if (binaryDependencies) {
                    assert(dependencies.size() == 1);
                }
                elem.first->setDependentEvents(dependencies);
                for (auto& dependency : dependencies) {
                    dependency->addIngoingDependency(elem.first);
                }
                
            }
            

            // Sort elements topologically
            // compute rank
            for (auto& elem : mElements) {
                computeRank(elem.second);
            }

            DFTElementVector elems = topoSort();
            size_t id = 0;
            for(DFTElementPointer e : elems) {
                e->setId(id++);
            }

            STORM_LOG_ASSERT(!mTopLevelIdentifier.empty(), "No top level element.");
            DFT<ValueType> dft(elems, mElements[mTopLevelIdentifier]);

            // Set layout info
            for (auto& elem : mElements) {
                if(mLayoutInfo.count(elem.first) > 0) {
                    dft.setElementLayoutInfo(elem.second->id(), mLayoutInfo.at(elem.first));
                } else {
                    // Set default layout
                    dft.setElementLayoutInfo(elem.second->id(), storm::storage::DFTLayoutInfo());
                }
            }

            return dft;
        }

        template<typename ValueType>
        unsigned DFTBuilder<ValueType>::computeRank(DFTElementPointer const& elem) {
            if(elem->rank() == static_cast<decltype(elem->rank())>(-1)) {
                if(elem->nrChildren() == 0 || elem->isDependency()) {
                    elem->setRank(0);
                } else {
                    DFTGatePointer gate = std::static_pointer_cast<DFTGate<ValueType>>(elem);
                    unsigned maxrnk = 0;
                    unsigned newrnk = 0;

                    for (DFTElementPointer const &child : gate->children()) {
                        newrnk = computeRank(child);
                        if (newrnk > maxrnk) {
                            maxrnk = newrnk;
                        }
                    }
                    elem->setRank(maxrnk + 1);
                }
            }

            return elem->rank();
        }

        template<typename ValueType>
        bool DFTBuilder<ValueType>::addRestriction(std::string const& name, std::vector<std::string> const& children, DFTElementType tp) {
            if (children.size() <= 1) {
                STORM_LOG_ERROR("Sequence enforcers require at least two children");
            }
            if (mElements.count(name) != 0) {
                return false;
            }
            DFTRestrictionPointer restr;
            switch (tp) {
                case DFTElementType::SEQ:
                    restr = std::make_shared<DFTSeq<ValueType>>(mNextId++, name);
                    break;
                case DFTElementType::MUTEX:
                    // TODO notice that mutex state generation support is lacking anyway, as DONT CARE propagation would be broken for this.
                    STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "Gate type not supported.");
                    break;
                default:
                    STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "Gate type not known.");
                    break;
            }

            mElements[name] = restr;
            mRestrictionChildNames[restr] = children;
            mRestrictions.push_back(restr);
            return true;
        }

        template<typename ValueType>
        bool DFTBuilder<ValueType>::addStandardGate(std::string const& name, std::vector<std::string> const& children, DFTElementType tp) {
            STORM_LOG_ASSERT(children.size() > 0, "No child.");
            if(mElements.count(name) != 0) {
                // Element with that name already exists.
                return false;
            }
            DFTElementPointer element;
            switch(tp) {
                case DFTElementType::AND:
                    element = std::make_shared<DFTAnd<ValueType>>(mNextId++, name);
                    break;
                case DFTElementType::OR:
                    element = std::make_shared<DFTOr<ValueType>>(mNextId++, name);
                    break;
                case DFTElementType::PAND:
                    element = std::make_shared<DFTPand<ValueType>>(mNextId++, name, pandDefaultInclusive);
                    break;
                case DFTElementType::POR:
                    element = std::make_shared<DFTPor<ValueType>>(mNextId++, name, porDefaultInclusive);
                    break;
                case DFTElementType::SPARE:
                   element = std::make_shared<DFTSpare<ValueType>>(mNextId++, name);
                   break;
                case DFTElementType::BE:
                case DFTElementType::VOT:
                case DFTElementType::PDEP:
                    // Handled separately
                    STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "Gate type handled separately.");
                case DFTElementType::CONSTF:
                case DFTElementType::CONSTS:
                    STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "Gate type not supported.");
                default:
                    STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "Gate type not known.");
            }
            mElements[name] = element;
            mChildNames[element] = children;
            return true;
        }

        template<typename ValueType>
        void DFTBuilder<ValueType>::topoVisit(DFTElementPointer const& n, std::map<DFTElementPointer, topoSortColour, OrderElementsById<ValueType>>& visited, DFTElementVector& L) {
            if(visited[n] == topoSortColour::GREY) {
                throw storm::exceptions::WrongFormatException("DFT is cyclic");
            } else if(visited[n] == topoSortColour::WHITE) {
                if(n->isGate()) {
                    visited[n] = topoSortColour::GREY;
                    for (DFTElementPointer const& c : std::static_pointer_cast<DFTGate<ValueType>>(n)->children()) {
                        topoVisit(c, visited, L);
                    }
                }
                // TODO restrictions and dependencies have no parents, so this can be done more efficiently.
                if(n->isRestriction()) {
                    visited[n] = topoSortColour::GREY;
                    for (DFTElementPointer const& c : std::static_pointer_cast<DFTRestriction<ValueType>>(n)->children()) {
                        topoVisit(c, visited, L);
                    }
                }
                if(n->isDependency()) {
                    visited[n] = topoSortColour::GREY;
                    for (DFTElementPointer const& c : std::static_pointer_cast<DFTDependency<ValueType>>(n)->dependentEvents()) {
                        topoVisit(c, visited, L);
                    }
                    topoVisit(std::static_pointer_cast<DFTDependency<ValueType>>(n)->triggerEvent(), visited, L);
                }
                visited[n] = topoSortColour::BLACK;
                L.push_back(n);
            }
        }

        template<typename ValueType>
        std::vector<std::shared_ptr<DFTElement<ValueType>>> DFTBuilder<ValueType>::topoSort() {
            std::map<DFTElementPointer, topoSortColour, OrderElementsById<ValueType>> visited;
            for(auto const& e : mElements) {
                visited.insert(std::make_pair(e.second, topoSortColour::WHITE)); 
            }

            DFTElementVector L;
            for(auto const& e : visited) {
                topoVisit(e.first, visited, L);
            }
            //std::reverse(L.begin(), L.end()); 
            return L;
        }
        
        template<typename ValueType>
        std::string DFTBuilder<ValueType>::getUniqueName(std::string name) {
            return name + "_" + std::to_string(++mUniqueOffset);
        }
        
        template<typename ValueType>
        void DFTBuilder<ValueType>::copyElement(DFTElementPointer element) {
            std::vector<std::string> children;
            switch (element->type()) {
                case DFTElementType::AND:
                case DFTElementType::OR:
                case DFTElementType::PAND:
                case DFTElementType::POR:
                case DFTElementType::SPARE:
                case DFTElementType::VOT:
                {
                    for (DFTElementPointer const& elem : std::static_pointer_cast<DFTGate<ValueType>>(element)->children()) {
                        children.push_back(elem->name());
                    }
                    copyGate(std::static_pointer_cast<DFTGate<ValueType>>(element), children);
                    break;
                }
                case DFTElementType::BE:
                {
                    std::shared_ptr<DFTBE<ValueType>> be = std::static_pointer_cast<DFTBE<ValueType>>(element);
                    ValueType dormancyFactor = storm::utility::zero<ValueType>();
                    if (be->canFail()) {
                        dormancyFactor = be->passiveFailureRate() / be->activeFailureRate();
                    }
                    addBasicElement(be->name(), be->activeFailureRate(), dormancyFactor);
                    break;
                }
                case DFTElementType::CONSTF:
                case DFTElementType::CONSTS:
                    // TODO
                    STORM_LOG_ASSERT(false, "Const elements not supported.");
                    break;
                case DFTElementType::PDEP:
                {
                    DFTDependencyPointer dependency = std::static_pointer_cast<DFTDependency<ValueType>>(element);
                    children.push_back(dependency->triggerEvent()->name());
                    for(auto const& depEv : dependency->dependentEvents()) {
                        children.push_back(depEv->name());
                    }
                    addDepElement(element->name(), children, dependency->probability());
                    break;
                }
                case DFTElementType::SEQ:
                case DFTElementType::MUTEX:
                {
                    for (DFTElementPointer const& elem : std::static_pointer_cast<DFTRestriction<ValueType>>(element)->children()) {
                        children.push_back(elem->name());
                    }
                    addRestriction(element->name(), children, element->type());
                    break;
                }
                default:
                    STORM_LOG_ASSERT(false, "Dft type not known.");
                    break;
            }
        }

        template<typename ValueType>
        void DFTBuilder<ValueType>::copyGate(DFTGatePointer gate, std::vector<std::string> const& children) {
            switch (gate->type()) {
                case DFTElementType::AND:
                case DFTElementType::OR:
                case DFTElementType::PAND:
                case DFTElementType::POR:
                case DFTElementType::SPARE:
                    addStandardGate(gate->name(), children, gate->type());
                    break;
                case DFTElementType::VOT:
                    addVotElement(gate->name(), std::static_pointer_cast<DFTVot<ValueType>>(gate)->threshold(), children);
                    break;
                default:
                    STORM_LOG_ASSERT(false, "Dft type not known.");
                    break;
            }
        }



        // Explicitly instantiate the class.
        template class DFTBuilder<double>;

#ifdef STORM_HAVE_CARL
        template class DFTBuilder<RationalFunction>;
#endif

    }
}
            
