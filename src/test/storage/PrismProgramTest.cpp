#include "gtest/gtest.h"
#include "storm-config.h"
#include "storm/parser/PrismParser.h"

#include "storm/utility/solver.h"

#include "storm/storage/jani/Model.h"

#ifdef STORM_HAVE_MSAT
TEST(PrismProgramTest, FlattenModules) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/leader3.nm"));

    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::MathsatSmtSolverFactory>();
    
    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(74, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Wlan_Mathsat) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/wlan0_collide.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::MathsatSmtSolverFactory>();

    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(179, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Csma_Mathsat) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/csma2_2.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::MathsatSmtSolverFactory>();

    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(70, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Firewire_Mathsat) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/firewire.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::MathsatSmtSolverFactory>();

    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(5024, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Coin_Mathsat) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/coin2.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::MathsatSmtSolverFactory>();

    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(13, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Dice_Mathsat) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/two_dice.nm"));

    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::MathsatSmtSolverFactory>();

    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(16, program.getModule(0).getNumberOfCommands());
}
#endif

#ifdef STORM_HAVE_Z3
TEST(PrismProgramTest, FlattenModules_Leader_Z3) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/leader3.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::Z3SmtSolverFactory>();
    
    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(74, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Wlan_Z3) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/wlan0_collide.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::Z3SmtSolverFactory>();
    
    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(179, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Csma_Z3) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/csma2_2.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::Z3SmtSolverFactory>();
    
    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(70, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Firewire_Z3) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/firewire.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::Z3SmtSolverFactory>();
    
    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(5024, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Coin_Z3) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/coin2.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::Z3SmtSolverFactory>();
    
    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(13, program.getModule(0).getNumberOfCommands());
}

TEST(PrismProgramTest, FlattenModules_Dice_Z3) {
    storm::prism::Program program;
    ASSERT_NO_THROW(program = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/two_dice.nm"));
    
    std::shared_ptr<storm::utility::solver::SmtSolverFactory> smtSolverFactory = std::make_shared<storm::utility::solver::Z3SmtSolverFactory>();
    
    ASSERT_NO_THROW(program = program.flattenModules(smtSolverFactory));
    EXPECT_EQ(1, program.getNumberOfModules());
    EXPECT_EQ(16, program.getModule(0).getNumberOfCommands());
}
#endif

TEST(PrismProgramTest, ConvertToJani) {
    storm::prism::Program prismProgram;
    storm::jani::Model janiModel;

    ASSERT_NO_THROW(prismProgram = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/leader3.nm"));
    ASSERT_NO_THROW(janiModel = prismProgram.toJani());
    
    ASSERT_NO_THROW(prismProgram = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/mdp/wlan0_collide.nm"));
    ASSERT_NO_THROW(janiModel = prismProgram.toJani());
    
    ASSERT_NO_THROW(prismProgram = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/dtmc/brp-16-2.pm"));
    ASSERT_NO_THROW(janiModel = prismProgram.toJani());

    ASSERT_NO_THROW(prismProgram = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/dtmc/crowds-5-5.pm"));
    ASSERT_NO_THROW(janiModel = prismProgram.toJani());
    
    ASSERT_NO_THROW(prismProgram = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/dtmc/leader-3-5.pm"));
    ASSERT_NO_THROW(janiModel = prismProgram.toJani());

    ASSERT_NO_THROW(prismProgram = storm::parser::PrismParser::parse(STORM_TEST_RESOURCES_DIR "/dtmc/nand-5-2.pm"));
    ASSERT_NO_THROW(janiModel = prismProgram.toJani());
}
