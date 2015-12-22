#include "src/storage/dd/cudd/InternalCuddDdManager.h"

#include "src/settings/SettingsManager.h"
#include "src/settings/modules/CuddSettings.h"

namespace storm {
    namespace dd {
        
        InternalDdManager<DdType::CUDD>::InternalDdManager() : cuddManager(), reorderingTechnique(CUDD_REORDER_NONE) {
            this->cuddManager.SetMaxMemory(static_cast<unsigned long>(storm::settings::cuddSettings().getMaximalMemory() * 1024ul * 1024ul));
            this->cuddManager.SetEpsilon(storm::settings::cuddSettings().getConstantPrecision());
            
            // Now set the selected reordering technique.
            storm::settings::modules::CuddSettings::ReorderingTechnique reorderingTechniqueAsSetting = storm::settings::cuddSettings().getReorderingTechnique();
            switch (reorderingTechniqueAsSetting) {
                case storm::settings::modules::CuddSettings::ReorderingTechnique::None: this->reorderingTechnique = CUDD_REORDER_NONE; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Random: this->reorderingTechnique = CUDD_REORDER_RANDOM; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::RandomPivot: this->reorderingTechnique = CUDD_REORDER_RANDOM_PIVOT; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Sift: this->reorderingTechnique = CUDD_REORDER_SIFT; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::SiftConv: this->reorderingTechnique = CUDD_REORDER_SIFT_CONVERGE; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::SymmetricSift: this->reorderingTechnique = CUDD_REORDER_SYMM_SIFT; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::SymmetricSiftConv: this->reorderingTechnique = CUDD_REORDER_SYMM_SIFT_CONV; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::GroupSift: this->reorderingTechnique = CUDD_REORDER_GROUP_SIFT; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::GroupSiftConv: this->reorderingTechnique = CUDD_REORDER_GROUP_SIFT_CONV; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Win2: this->reorderingTechnique = CUDD_REORDER_WINDOW2; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Win2Conv: this->reorderingTechnique = CUDD_REORDER_WINDOW2_CONV; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Win3: this->reorderingTechnique = CUDD_REORDER_WINDOW3; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Win3Conv: this->reorderingTechnique = CUDD_REORDER_WINDOW3_CONV; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Win4: this->reorderingTechnique = CUDD_REORDER_WINDOW4; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Win4Conv: this->reorderingTechnique = CUDD_REORDER_WINDOW4_CONV; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Annealing: this->reorderingTechnique = CUDD_REORDER_ANNEALING; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Genetic: this->reorderingTechnique = CUDD_REORDER_GENETIC; break;
                case storm::settings::modules::CuddSettings::ReorderingTechnique::Exact: this->reorderingTechnique = CUDD_REORDER_EXACT; break;
            }
        }
        
        InternalDdManager<DdType::CUDD>::~InternalDdManager() {
            // Intentionally left empty.
        }
        
        InternalBdd<DdType::CUDD> InternalDdManager<DdType::CUDD>::getBddOne() const {
            return InternalBdd<DdType::CUDD>(this, cuddManager.bddOne());
        }
        
        template<typename ValueType>
        InternalAdd<DdType::CUDD, ValueType> InternalDdManager<DdType::CUDD>::getAddOne() const {
            return InternalAdd<DdType::CUDD, ValueType>(this, cuddManager.addOne());
        }
        
        InternalBdd<DdType::CUDD> InternalDdManager<DdType::CUDD>::getBddZero() const {
            return InternalBdd<DdType::CUDD>(this, cuddManager.bddZero());
        }
        
        template<typename ValueType>
        InternalAdd<DdType::CUDD, ValueType> InternalDdManager<DdType::CUDD>::getAddZero() const {
            return InternalAdd<DdType::CUDD, ValueType>(this, cuddManager.addZero());
        }
        
        template<typename ValueType>
        InternalAdd<DdType::CUDD, ValueType> InternalDdManager<DdType::CUDD>::getConstant(ValueType const& value) const {
            return InternalAdd<DdType::CUDD, ValueType>(this, cuddManager.constant(value));
        }

        std::pair<InternalBdd<DdType::CUDD>, InternalBdd<DdType::CUDD>> InternalDdManager<DdType::CUDD>::createNewDdVariablePair() {
            std::pair<InternalBdd<DdType::CUDD>, InternalBdd<DdType::CUDD>> result;
            result.first = InternalBdd<DdType::CUDD>(this, cuddManager.bddVar());
            result.second = InternalBdd<DdType::CUDD>(this, cuddManager.bddVar());
            
            // Connect the two variables so they are not 'torn apart' during dynamic reordering.
            cuddManager.MakeTreeNode(result.first.getIndex(), 2, MTR_FIXED);
            
            return result;
        }
        
        void InternalDdManager<DdType::CUDD>::allowDynamicReordering(bool value) {
            if (value) {
                this->getCuddManager().AutodynEnable(this->reorderingTechnique);
            } else {
                this->getCuddManager().AutodynDisable();
            }
        }
        
        bool InternalDdManager<DdType::CUDD>::isDynamicReorderingAllowed() const {
            Cudd_ReorderingType type;
            return this->getCuddManager().ReorderingStatus(&type);
        }
        
        void InternalDdManager<DdType::CUDD>::triggerReordering() {
            this->getCuddManager().ReduceHeap(this->reorderingTechnique, 0);
        }
        
        cudd::Cudd& InternalDdManager<DdType::CUDD>::getCuddManager() {
            return cuddManager;
        }
        
        cudd::Cudd const& InternalDdManager<DdType::CUDD>::getCuddManager() const {
            return cuddManager;
        }

        template InternalAdd<DdType::CUDD, double> InternalDdManager<DdType::CUDD>::getAddOne() const;
        template InternalAdd<DdType::CUDD, uint_fast64_t> InternalDdManager<DdType::CUDD>::getAddOne() const;
        
        template InternalAdd<DdType::CUDD, double> InternalDdManager<DdType::CUDD>::getAddZero() const;
        template InternalAdd<DdType::CUDD, uint_fast64_t> InternalDdManager<DdType::CUDD>::getAddZero() const;
        
        template InternalAdd<DdType::CUDD, double> InternalDdManager<DdType::CUDD>::getConstant(double const& value) const;
        template InternalAdd<DdType::CUDD, uint_fast64_t> InternalDdManager<DdType::CUDD>::getConstant(uint_fast64_t const& value) const;
    }
}