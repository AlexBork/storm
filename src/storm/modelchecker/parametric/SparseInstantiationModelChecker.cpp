#include "SparseInstantiationModelChecker.h"

#include "storm/adapters/CarlAdapter.h"
#include "storm/models/sparse/Dtmc.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"

#include "storm/exceptions/InvalidArgumentException.h"

namespace storm {
    namespace modelchecker {
        namespace parametric {
            
            template <typename SparseModelType, typename ConstantType>
            SparseInstantiationModelChecker<SparseModelType, ConstantType>::SparseInstantiationModelChecker(SparseModelType const& parametricModel) : parametricModel(parametricModel) {
                //Intentionally left empty
            }
            
                
            template <typename SparseModelType, typename ConstantType>
            void SparseInstantiationModelChecker<SparseModelType, ConstantType>::specifyFormula(storm::modelchecker::CheckTask<storm::logic::Formula, typename SparseModelType::ValueType> const& checkTask) {
                currentFormula = checkTask.getFormula().asSharedPointer();
                currentCheckTask = std::make_unique<storm::modelchecker::CheckTask<storm::logic::Formula, ConstantType>>(*currentFormula, checkTask.isOnlyInitialStatesRelevantSet());
                currentCheckTask->setProduceSchedulers(checkTask.isProduceSchedulersSet());
            }
            
            template class SparseInstantiationModelChecker<storm::models::sparse::Dtmc<storm::RationalFunction>, double>;
            template class SparseInstantiationModelChecker<storm::models::sparse::Mdp<storm::RationalFunction>, double>;

        }
    }
}