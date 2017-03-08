#include "gtest/gtest.h"
#include "storm-config.h"

#ifdef STORM_HAVE_CARL

#include "storm/adapters/CarlAdapter.h"

#include "utility/storm.h"
#include "storm/models/sparse/Model.h"
#include "modelchecker/parametric/ParameterLifting.h"

TEST(SparseDtmcRegionModelCheckerTest, Brp_Prob) {
    
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/brp16_2.pm";
    std::string formulaAsString = "P<=0.84 [F s=5 ]";
    std::string constantsAsString = ""; //e.g. pL=0.9,TOACK=0.5

    // Program and formula
    storm::prism::Program program = storm::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::extractFormulasFromProperties(storm::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    
    storm::modelchecker::parametric::ParameterLifting<storm::models::sparse::Dtmc<storm::RationalFunction>, double> parameterLiftingContext(model);
    parameterLiftingContext.specifyFormula(*formulas[0]);

    //start testing
    auto allSatRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.7<=pL<=0.9,0.75<=pK<=0.95");
    auto exBothRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.4<=pL<=0.65,0.75<=pK<=0.95");
    auto allVioRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.1<=pL<=0.73,0.2<=pK<=0.715");

    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllSat, parameterLiftingContext.analyzeRegion(allSatRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::ExistsBoth, parameterLiftingContext.analyzeRegion(exBothRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllViolated, parameterLiftingContext.analyzeRegion(allVioRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));

    carl::VariablePool::getInstance().clear();
}

TEST(SparseDtmcRegionModelCheckerTest, Brp_Rew) {
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/brp_rewards16_2.pm";
    std::string formulaAsString = "R>2.5 [F ((s=5) | (s=0&srep=3)) ]";
    std::string constantsAsString = "pL=0.9,TOAck=0.5";
    carl::VariablePool::getInstance().clear();
    
    storm::prism::Program program = storm::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::extractFormulasFromProperties(storm::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    
    storm::modelchecker::parametric::ParameterLifting<storm::models::sparse::Dtmc<storm::RationalFunction>, double> parameterLiftingContext(model);
    parameterLiftingContext.specifyFormula(*formulas[0]);
    
    //start testing
    auto allSatRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.7<=pK<=0.875,0.75<=TOMsg<=0.95");
    auto exBothRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.6<=pK<=0.9,0.5<=TOMsg<=0.95");
    auto exBothHardRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.5<=pK<=0.75,0.3<=TOMsg<=0.4"); //this region has a local maximum!
    auto allVioRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.1<=pK<=0.3,0.2<=TOMsg<=0.3");

    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllSat, parameterLiftingContext.analyzeRegion(allSatRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::ExistsBoth, parameterLiftingContext.analyzeRegion(exBothRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllViolated, parameterLiftingContext.analyzeRegion(allVioRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    
    carl::VariablePool::getInstance().clear();

}

TEST(SparseDtmcRegionModelCheckerTest, Brp_Rew_Infty) {
    
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/brp_rewards16_2.pm";
    std::string formulaAsString = "R>2.5 [F (s=0&srep=3) ]";
    std::string constantsAsString = "";
    carl::VariablePool::getInstance().clear();
    storm::prism::Program program = storm::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::extractFormulasFromProperties(storm::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    
    storm::modelchecker::parametric::ParameterLifting<storm::models::sparse::Dtmc<storm::RationalFunction>, double> parameterLiftingContext(model);
    parameterLiftingContext.specifyFormula(*formulas[0]);

    //start testing
    auto allSatRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("");
    
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllSat, parameterLiftingContext.analyzeRegion(allSatRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));

    carl::VariablePool::getInstance().clear();
}

TEST(SparseDtmcRegionModelCheckerTest, Brp_Rew_4Par) {
    
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/brp_rewards16_2.pm";
    std::string formulaAsString = "R>2.5 [F ((s=5) | (s=0&srep=3)) ]";
    std::string constantsAsString = ""; //!! this model will have 4 parameters
    carl::VariablePool::getInstance().clear();
    storm::prism::Program program = storm::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::extractFormulasFromProperties(storm::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    
    storm::modelchecker::parametric::ParameterLifting<storm::models::sparse::Dtmc<storm::RationalFunction>, double> parameterLiftingContext(model);
    parameterLiftingContext.specifyFormula(formulas[0]);
    
    //start testing
    auto allSatRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.7<=pK<=0.9,0.6<=pL<=0.85,0.9<=TOMsg<=0.95,0.85<=TOAck<=0.9");
    auto exBothRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.1<=pK<=0.7,0.2<=pL<=0.8,0.15<=TOMsg<=0.65,0.3<=TOAck<=0.9");
    auto allVioRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.1<=pK<=0.4,0.2<=pL<=0.3,0.15<=TOMsg<=0.3,0.1<=TOAck<=0.2");

    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllSat, parameterLiftingContext.analyzeRegion(allSatRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::ExistsBoth, parameterLiftingContext.analyzeRegion(exBothRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllViolated, parameterLiftingContext.analyzeRegion(allVioRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    
    carl::VariablePool::getInstance().clear();
}

TEST(SparseDtmcRegionModelCheckerTest, Crowds_Prob) {
    
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/crowds3_5.pm";
    std::string formulaAsString = "P<0.5 [F \"observe0Greater1\" ]";
    std::string constantsAsString = ""; //e.g. pL=0.9,TOACK=0.5
    carl::VariablePool::getInstance().clear();

    storm::prism::Program program = storm::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::extractFormulasFromProperties(storm::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
   
    storm::modelchecker::parametric::ParameterLifting<storm::models::sparse::Dtmc<storm::RationalFunction>, double> parameterLiftingContext(model);
    parameterLiftingContext.specifyFormula(formulas[0]);

    //start testing
    auto allSatRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.1<=PF<=0.75,0.15<=badC<=0.2");
    auto exBothRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.75<=PF<=0.8,0.2<=badC<=0.3");
    auto allVioRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.8<=PF<=0.95,0.2<=badC<=0.2");
    auto allVioHardRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.8<=PF<=0.95,0.2<=badC<=0.9");

    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllSat, parameterLiftingContext.analyzeRegion(allSatRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::ExistsBoth, parameterLiftingContext.analyzeRegion(exBothRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllViolated, parameterLiftingContext.analyzeRegion(allVioRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::CenterViolated, parameterLiftingContext.analyzeRegion(allVioHardRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));

    carl::VariablePool::getInstance().clear();
}

TEST(SparseDtmcRegionModelCheckerTest, Crowds_Prob_1Par) {
    
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/crowds3_5.pm";
    std::string formulaAsString = "P>0.75 [F \"observe0Greater1\" ]";
    std::string constantsAsString = "badC=0.3"; //e.g. pL=0.9,TOACK=0.5
    carl::VariablePool::getInstance().clear();

    
    storm::prism::Program program = storm::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::extractFormulasFromProperties(storm::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    
    storm::modelchecker::parametric::ParameterLifting<storm::models::sparse::Dtmc<storm::RationalFunction>, double> parameterLiftingContext(model);
    parameterLiftingContext.specifyFormula(formulas[0]);
    
    //start testing
    auto allSatRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.9<=PF<=0.99");
    auto exBothRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.8<=PF<=0.9");
    auto allVioRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("0.01<=PF<=0.8");

    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllSat, parameterLiftingContext.analyzeRegion(allSatRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::ExistsBoth, parameterLiftingContext.analyzeRegion(exBothRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllViolated, parameterLiftingContext.analyzeRegion(allVioRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));
    
    carl::VariablePool::getInstance().clear();
}

TEST(SparseDtmcRegionModelCheckerTest, Crowds_Prob_Const) {
    
    std::string programFile = STORM_TEST_RESOURCES_DIR "/pdtmc/crowds3_5.pm";
    std::string formulaAsString = "P>0.6 [F \"observe0Greater1\" ]";
    std::string constantsAsString = "PF=0.9,badC=0.2";
    carl::VariablePool::getInstance().clear();
    
    storm::prism::Program program = storm::parseProgram(programFile);
    program = storm::utility::prism::preprocess(program, constantsAsString);
    std::vector<std::shared_ptr<const storm::logic::Formula>> formulas = storm::extractFormulasFromProperties(storm::parsePropertiesForPrismProgram(formulaAsString, program));
    std::shared_ptr<storm::models::sparse::Dtmc<storm::RationalFunction>> model = storm::buildSparseModel<storm::RationalFunction>(program, formulas)->as<storm::models::sparse::Dtmc<storm::RationalFunction>>();
    
    storm::modelchecker::parametric::ParameterLifting<storm::models::sparse::Dtmc<storm::RationalFunction>, double> parameterLiftingContext(model);
    parameterLiftingContext.specifyFormula(formulas[0]);

    //start testing
    auto allSatRegion=storm::storage::ParameterRegion<storm::RationalFunction>::parseRegion("");
    
    EXPECT_EQ(storm::modelchecker::parametric::RegionCheckResult::AllSat, parameterLiftingContext.analyzeRegion(allSatRegion, storm::modelchecker::parametric::RegionCheckResult::Unknown, true));

    carl::VariablePool::getInstance().clear();
}

#endif
