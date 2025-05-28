import time
import statistics
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os

from minions.clients.openai import OpenAIClient
from minions.clients.lemonade import LemonadeClient
from minions.minion import Minion

class LemonadePerformanceTester:
    def __init__(self, models, openai_api_key: str = None):
        """Initialize the performance tester with available Lemonade models."""
        # Try to get OpenAI API key from environment variables first
        # If not found, use the provided key parameter
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Available Lemonade models from your configuration
        self.lemonade_models = models
        
        # Test prompts of different sizes
        self.test_prompts = {
            "100_words": self._generate_prompt(100)
            #"500_words": self._generate_prompt(500), 
            #"1000_words": self._generate_prompt(2000)
        }
        
        # Base medical context for consistency
        self.base_context = """
        Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, 
        his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort 
        during physical activity. Recent laboratory results show that his LDL cholesterol level is 
        elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other 
        metabolic indicators, including fasting glucose and renal function, are unremarkable.
        """

    def _generate_prompt(self, target_words: int) -> str:
        """Generate prompts of approximately the target word count."""
        if target_words <= 100:
            return """
            Based on the patient's blood pressure readings of 160/100 mmHg and elevated LDL cholesterol 
            at 170 mg/dL, provide a comprehensive cardiovascular risk assessment. Consider the patient's 
            age, gender, and reported symptoms of chest discomfort during physical activity. Evaluate 
            whether these factors together suggest an increased risk for cardiovascular complications 
            such as myocardial infarction or stroke. Additionally, recommend appropriate lifestyle 
            modifications, potential medication adjustments, and follow-up care strategies. Discuss 
            the significance of the normal HDL levels and unremarkable glucose and renal function 
            in the overall risk profile. Provide evidence-based treatment recommendations following 
            current cardiovascular disease prevention guidelines.
            """
        
        elif target_words <= 500:
            extended_context = self.base_context + """
            
            Additional patient history reveals that John Doe has been experiencing these symptoms 
            for approximately 6 months. His family history is significant for coronary artery disease, 
            with his father having suffered a myocardial infarction at age 65 and his mother having 
            a history of stroke at age 70. The patient is currently employed as a desk worker with 
            minimal physical activity throughout the day. His diet consists primarily of processed 
            foods high in sodium and saturated fats. He consumes alcohol socially, approximately 
            2-3 drinks per week, and has never smoked tobacco products.
            
            Current medications include lisinopril 10mg daily for hypertension, which was started 
            3 years ago. His blood pressure has been gradually increasing despite medication compliance. 
            The patient reports good adherence to his medication regimen and denies any side effects. 
            Recent echocardiogram results show normal left ventricular function with an ejection 
            fraction of 60%. However, there are mild signs of left ventricular hypertrophy, which 
            is consistent with long-standing hypertension.
            
            Laboratory workup also reveals a hemoglobin A1c of 5.8%, indicating prediabetic glucose 
            metabolism. Thyroid function tests are within normal limits. Complete blood count shows 
            no abnormalities. Comprehensive metabolic panel reveals normal electrolyte levels and 
            kidney function with a creatinine of 1.0 mg/dL and estimated GFR of 85 mL/min/1.73m².
            
            Physical examination findings include a BMI of 28.5 kg/m², indicating overweight status. 
            Cardiovascular examination reveals a regular rate and rhythm with no murmurs, gallops, 
            or rubs. Peripheral pulses are intact bilaterally. Fundoscopic examination shows mild 
            arteriovenous nicking consistent with hypertensive retinopathy. The patient's exercise 
            tolerance is limited, reporting dyspnea and chest discomfort with moderate exertion 
            such as climbing two flights of stairs.
            """
            
            return f"""
            Provide a comprehensive cardiovascular risk stratification and management plan for this 
            60-year-old male patient. Your analysis should include: 1) Detailed assessment of current 
            cardiovascular risk factors including hypertension, dyslipidemia, family history, and 
            lifestyle factors. 2) Calculation of 10-year cardiovascular risk using appropriate risk 
            calculators such as the Pooled Cohort Equations. 3) Evaluation of the significance of 
            chest discomfort symptoms and recommendations for further cardiac evaluation including 
            stress testing or cardiac imaging if indicated. 4) Comprehensive treatment plan addressing 
            both pharmacological and non-pharmacological interventions. 5) Specific recommendations 
            for blood pressure management including potential medication adjustments or additions. 
            6) Lipid management strategies including statin therapy considerations and target LDL goals. 
            7) Lifestyle modification recommendations including dietary counseling, exercise prescription, 
            and weight management strategies. 8) Monitoring plan with specific follow-up intervals and 
            laboratory assessments. 9) Patient education priorities and shared decision-making approaches. 
            10) Consideration of emerging cardiovascular risk factors and novel therapeutic approaches. 
            Please provide evidence-based recommendations citing current guidelines from major cardiovascular 
            societies and recent clinical trial data where applicable. Context: {extended_context}
            """
        
        else:  # 1000 words
            return """Provide a comprehensive cardiovascular risk stratification and management plan for this 
            60-year-old male patient. Your analysis should include: 1) Detailed assessment of current cardiovascular 
            risk factors including hypertension, dyslipidemia, family history, and lifestyle factors. 2) Calculation 
            of 10-year cardiovascular risk using appropriate risk calculators such as the Pooled Cohort Equations or 
            the newer PREVENT risk assessment tool which incorporates kidney function, blood sugar levels, and social 
            determinants of health. 3) Evaluation of the significance of chest discomfort symptoms and recommendations 
            for further cardiac evaluation including stress testing or cardiac imaging if indicated. 4) Comprehensive 
            treatment plan addressing both pharmacological and non-pharmacological interventions. 5) Specific 
            recommendations for blood pressure management including potential medication adjustments or additions 
            considering current lisinopril therapy and gradual blood pressure increases despite medication compliance. 
            6) Lipid management strategies including statin therapy considerations and target LDL goals based on current
            guidelines from major cardiovascular societies. 7) Lifestyle modification recommendations including 
            dietary counseling, exercise prescription, and weight management strategies addressing the patient's 
            sedentary occupation and current BMI of 28.5 kg/m². 8) Monitoring plan with specific follow-up intervals 
            and laboratory assessments. 9) Patient education priorities and shared decision-making approaches incorporating 
            visual aids such as heart age and risk trajectory tools. 10) Consideration of emerging cardiovascular 
            risk factors and novel therapeutic approaches. Please provide evidence-based recommendations citing 
            current guidelines from major cardiovascular societies and recent clinical trial data where applicable. 
            Additionally, analyze the patient's current medication regimen for potential drug interactions, 
            contraindications, and opportunities for optimization. Consider the role of aspirin therapy for primary 
            prevention given his risk profile and bleeding risk assessment. Evaluate the need for additional cardiovascular 
            biomarkers such as high-sensitivity C-reactive protein, lipoprotein(a), coronary artery calcium scoring, or 
            advanced lipid testing to further refine risk assessment. Discuss the implications of his prediabetic glucose 
            metabolism with hemoglobin A1c of 5.8% and provide comprehensive recommendations for diabetes prevention 
            strategies including metformin consideration. Address the significance of his occupational sedentary lifestyle 
            and provide specific workplace wellness interventions and ergonomic assessments. Consider the psychosocial 
            aspects of cardiovascular disease prevention including stress management techniques, sleep hygiene 
            optimization, and mental health screening for depression and anxiety. Evaluate the role of family 
            screening given his significant family history of coronary artery disease and stroke with recommendations 
            for genetic counseling if appropriate. Provide detailed recommendations for emergency action planning and 
            when to seek immediate medical attention including recognition of acute coronary syndrome symptoms. Discuss 
            the cost-effectiveness of various interventions and consider patient preferences in treatment selection while 
            addressing potential barriers to adherence and strategies to improve medication compliance and lifestyle 
            modification sustainability. Consider the role of digital health tools, remote monitoring technologies, and 
            telemedicine in ongoing care management. Evaluate the need for subspecialty referrals including cardiology for 
            advanced risk stratification, endocrinology for prediabetes management, or nutrition counseling for comprehensive
            dietary intervention. Discuss the timing and frequency of follow-up visits and laboratory monitoring including 
            lipid panels, hemoglobin A1c, kidney function, and liver function tests. Address the role of cardiac 
            rehabilitation if indicated and community resources for lifestyle modification support. Consider age-specific 
            cardiovascular risk factors and screening recommendations including evaluation for subclinical atherosclerosis.
            Evaluate the need for additional imaging studies such as carotid ultrasound for intima-media thickness 
            measurement, ankle-brachial index for peripheral arterial disease screening, or echocardiography for left 
            ventricular hypertrophy assessment. Discuss the role of genetic testing for familial hypercholesterolemia 
            or other inherited cardiovascular conditions given the strong family history. Address vaccination 
            recommendations including annual influenza and pneumococcal vaccines as part of comprehensive cardiovascular
            risk reduction strategies. Consider the impact of seasonal variations on cardiovascular risk and provide 
            appropriate counseling for winter months and holiday periods. Evaluate the role of supplements and 
            nutraceuticals in cardiovascular disease prevention including omega-3 fatty acids, coenzyme Q10, and plant 
            sterols. Discuss the importance of dental health and periodontal disease management in cardiovascular 
            risk reduction with recommendations for regular dental care. Address travel considerations and medication 
            management during travel including time zone adjustments and emergency medication supplies. Consider the 
            role of cardiac screening for competitive sports or high-intensity exercise participation given his current 
            exercise intolerance. Evaluate the need for workplace accommodations or restrictions based on cardiovascular
            risk assessment and occupational health considerations. Discuss the role of family involvement in lifestyle
            modification and medication adherence including spouse and family member education. Address cultural and 
            socioeconomic factors that may impact treatment recommendations and adherence including health literacy 
            assessment. Consider the role of community health programs and resources for ongoing support including 
            cardiac rehabilitation programs and support groups. Evaluate the need for advance directives and end-of-life 
            planning discussions as part of comprehensive care. Discuss the importance of regular self-monitoring including 
            home blood pressure monitoring with proper technique training and weight tracking. Address the role of alcohol 
            consumption in cardiovascular health and provide appropriate counseling regarding his current social drinking 
            patterns. Consider the impact of environmental factors such as air pollution exposure and noise exposure on 
            cardiovascular risk with mitigation strategies. Evaluate the need for sleep study evaluation given potential 
            cardiovascular implications of sleep disorders. Discuss the role of mindfulness, meditation, and stress reduction 
            techniques in cardiovascular health including specific programs and resources. Address the importance of social 
            support networks in maintaining healthy lifestyle behaviors and cardiovascular health outcomes. Consider the role 
            of pet ownership and social connections in cardiovascular health outcomes and overall well-being. Evaluate the 
            need for occupational health assessments and workplace safety considerations related to cardiovascular health. 
            Discuss the importance of regular dental care and oral health maintenance as part of cardiovascular disease 
            prevention. Address the role of seasonal affective disorder and vitamin D status in cardiovascular health with 
            appropriate screening and supplementation recommendations. Consider the impact of shift work or irregular sleep 
            schedules on cardiovascular risk if applicable to his work situation. Evaluate the need for hormone replacement 
            therapy considerations if applicable and discuss cardiovascular implications. Discuss the role of inflammatory 
            markers and autoimmune conditions in cardiovascular risk assessment including screening recommendations. Address 
            the importance of cancer screening and the cardiovascular implications of cancer treatments if relevant. Consider 
            the role of kidney function monitoring and nephrology consultation if indicated based on current creatinine levels.
            Evaluate the need for thyroid function assessment and endocrine evaluation as part of comprehensive metabolic 
            assessment. Discuss the importance of bone health and the cardiovascular implications of osteoporosis medications
            if relevant. Address the role of cognitive function assessment and the relationship between cardiovascular 
            and brain health including vascular dementia prevention."""
            


    def test_model_performance(
        self, 
        model_name: str, 
        prompt_size: str, 
        iterations: int = 5
    ) -> Dict:
        """Test a specific model with a specific prompt size over multiple iterations."""
        
        print(f"\n{'='*80}")
        print(f"Testing {model_name} with {prompt_size} prompt")
        print(f"{'='*80}")
        
        results = {
            "model_name": model_name,
            "prompt_size": prompt_size,
            "iterations": iterations,
            "iteration_details": [],
            "errors": []
        }
        
        for i in range(iterations):
            try:
                print(f"\n--- Iteration {i+1}/{iterations} ---")
                
                # Initialize clients
                local_client = LemonadeClient(model_name=model_name)
                remote_client = OpenAIClient(
                    model_name="gpt-4o",
                    api_key=self.openai_api_key
                )
                
                # Create Minion instance
                minion = Minion(local_client, remote_client)
                
                # Initialize timing variables (following app.py pattern)
                local_time_spent = 0
                remote_time_spent = 0
                
                # Store original chat methods (following app.py pattern)
                original_local_chat = local_client.chat
                original_remote_chat = remote_client.chat
                
                # Create timing wrapper for local client (following app.py pattern)
                def timed_local_chat(*args, **kwargs):
                    nonlocal local_time_spent
                    start_time = time.time()
                    result = original_local_chat(*args, **kwargs)
                    local_time_spent += time.time() - start_time
                    return result
                
                # Create timing wrapper for remote client (following app.py pattern)
                def timed_remote_chat(*args, **kwargs):
                    nonlocal remote_time_spent
                    start_time = time.time()
                    result = original_remote_chat(*args, **kwargs)
                    remote_time_spent += time.time() - start_time
                    return result
                
                # Replace the chat methods with the timed versions (following app.py pattern)
                local_client.chat = timed_local_chat
                remote_client.chat = timed_remote_chat
                
                # Record start time for total execution
                execution_start_time = time.time()
                
                # Execute the task
                output = minion(
                    task=self.test_prompts[prompt_size],
                    context=[self.base_context],
                    max_rounds=2
                )
                
                # Calculate total execution time
                execution_time = time.time() - execution_start_time
                
                # Restore original chat methods (following app.py pattern)
                local_client.chat = original_local_chat
                remote_client.chat = original_remote_chat
                
                # Extract usage information from output (following app.py pattern)
                local_usage = output.get("local_usage", {})
                remote_usage = output.get("remote_usage", {})
                
                # Calculate token totals
                local_total_tokens = local_usage.get("prompt_tokens", 0) + local_usage.get("completion_tokens", 0)
                remote_total_tokens = remote_usage.get("prompt_tokens", 0) + remote_usage.get("completion_tokens", 0)
                
                # Calculate other time (overhead) following app.py pattern
                other_time = execution_time - (local_time_spent + remote_time_spent)
                
                # Store detailed iteration results
                iteration_detail = {
                    "iteration": i + 1,
                    "local_time": local_time_spent,
                    "remote_time": remote_time_spent,
                    "execution_time": execution_time,
                    "other_time": other_time,
                    "local_usage": local_usage,
                    "remote_usage": remote_usage,
                    "local_total_tokens": local_total_tokens,
                    "remote_total_tokens": remote_total_tokens
                }
                
                results["iteration_details"].append(iteration_detail)
                
                # Print detailed results for this iteration
                print(f"⏱️  Processing Times:")
                print(f"   Local (Lemonade):  {local_time_spent:.3f}s")
                print(f"   Remote (OpenAI):   {remote_time_spent:.3f}s")
                print(f"   Execution Time:    {execution_time:.3f}s")
                print(f"   Overhead:          {other_time:.3f}s")
                
                print(f"\n🔢 Token Usage:")
                print(f"   Local Model:")
                print(f"     Prompt tokens:     {local_usage.get('prompt_tokens', 0):,}")
                print(f"     Completion tokens: {local_usage.get('completion_tokens', 0):,}")
                print(f"     Total tokens:      {local_total_tokens:,}")
                
                print(f"   Remote Model:")
                print(f"     Prompt tokens:     {remote_usage.get('prompt_tokens', 0):,}")
                print(f"     Completion tokens: {remote_usage.get('completion_tokens', 0):,}")
                print(f"     Total tokens:      {remote_total_tokens:,}")
                
                print(f"\n📊 Performance Metrics:")
                if local_time_spent > 0:
                    local_tokens_per_sec = local_total_tokens / local_time_spent
                    print(f"   Local throughput:  {local_tokens_per_sec:.1f} tokens/sec")
                if remote_time_spent > 0:
                    remote_tokens_per_sec = remote_total_tokens / remote_time_spent
                    print(f"   Remote throughput: {remote_tokens_per_sec:.1f} tokens/sec")
                
                # Brief response preview
                final_answer = output.get("final_answer", "")
                preview = final_answer[:100] + "..." if len(final_answer) > 100 else final_answer
                print(f"\n💬 Response Preview: {preview}")
                
            except Exception as e:
                error_msg = f"Iteration {i+1} failed: {str(e)}"
                print(f"❌ ERROR: {error_msg}")
                results["errors"].append(error_msg)
        
        # Print summary statistics for this model/prompt combination
        if results["iteration_details"]:
            self._print_iteration_summary(results)
        
        return results

    def _print_iteration_summary(self, results: Dict):
        """Print summary statistics for a completed test."""
        details = results["iteration_details"]
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {results['model_name']} - {results['prompt_size']}")
        print(f"{'='*60}")
        
        # Calculate averages
        local_times = [d["local_time"] for d in details]
        remote_times = [d["remote_time"] for d in details]
        execution_times = [d["execution_time"] for d in details]
        other_times = [d["other_time"] for d in details]
        local_tokens = [d["local_total_tokens"] for d in details]
        remote_tokens = [d["remote_total_tokens"] for d in details]
        
        print(f"📈 Average Performance ({len(details)} successful iterations):")
        print(f"   Local time:       {statistics.mean(local_times):.3f}s (±{statistics.stdev(local_times) if len(local_times) > 1 else 0:.3f})")
        print(f"   Remote time:      {statistics.mean(remote_times):.3f}s (±{statistics.stdev(remote_times) if len(remote_times) > 1 else 0:.3f})")
        print(f"   Execution time:   {statistics.mean(execution_times):.3f}s (±{statistics.stdev(execution_times) if len(execution_times) > 1 else 0:.3f})")
        print(f"   Overhead time:    {statistics.mean(other_times):.3f}s (±{statistics.stdev(other_times) if len(other_times) > 1 else 0:.3f})")
        print(f"   Local tokens:     {statistics.mean(local_tokens):.0f} (±{statistics.stdev(local_tokens) if len(local_tokens) > 1 else 0:.0f})")
        print(f"   Remote tokens:    {statistics.mean(remote_tokens):.0f} (±{statistics.stdev(remote_tokens) if len(remote_tokens) > 1 else 0:.0f})")
        
        if results["errors"]:
            print(f"\n❌ Failed iterations: {len(results['errors'])}")
            for error in results["errors"]:
                print(f"   - {error}")

    def run_comprehensive_test(self, iterations: int = 5) -> Dict:
        """Run comprehensive tests across all models and prompt sizes."""
        
        print("🚀 Starting comprehensive Lemonade performance testing...")
        print(f"📋 Testing {len(self.lemonade_models)} models with {len(self.test_prompts)} prompt sizes")
        print(f"🔄 Running {iterations} iterations per test")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        all_results = []
        
        for model_idx, model in enumerate(self.lemonade_models):
            print(f"\n🤖 Model {model_idx + 1}/{len(self.lemonade_models)}: {model}")
            
            for prompt_idx, prompt_size in enumerate(self.test_prompts.keys()):
                print(f"📝 Prompt {prompt_idx + 1}/{len(self.test_prompts)}: {prompt_size}")
                
                try:
                    # Test this model/prompt combination
                    results = self.test_model_performance(model, prompt_size, iterations)
                    all_results.append(results)
                    
                except Exception as e:
                    print(f"❌ Failed to test {model} with {prompt_size}: {str(e)}")
        
        return {
            "test_timestamp": datetime.now().isoformat(),
            "detailed_results": all_results
        }

    def save_results(self, test_results: Dict, filename: str = None):
        """Save test results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lemonade_performance_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")

    def print_final_summary(self, test_results: Dict):
        """Print a comprehensive final summary."""
        print(f"\n{'='*80}")
        print("🏁 FINAL PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        results = test_results["detailed_results"]
        
        # Group by prompt size for comparison
        for prompt_size in self.test_prompts.keys():
            print(f"\n📊 {prompt_size.upper()} PROMPT RESULTS:")
            print("-" * 60)
            
            size_results = [r for r in results if r["prompt_size"] == prompt_size and r["iteration_details"]]
            
            if not size_results:
                print("   No successful results for this prompt size")
                continue
            
            # Sort by average execution time
            size_results.sort(key=lambda x: statistics.mean([d["execution_time"] for d in x["iteration_details"]]))
            
            print(f"{'Model':<40} {'Avg Exec Time':<15} {'Avg Local/Remote':<20} {'Avg Tokens':<15}")
            print("-" * 90)
            
            for result in size_results:
                details = result["iteration_details"]
                model_name = result["model_name"][:37] + "..." if len(result["model_name"]) > 37 else result["model_name"]
                
                avg_exec_time = statistics.mean([d["execution_time"] for d in details])
                avg_local_time = statistics.mean([d["local_time"] for d in details])
                avg_remote_time = statistics.mean([d["remote_time"] for d in details])
                avg_local_tokens = statistics.mean([d["local_total_tokens"] for d in details])
                avg_remote_tokens = statistics.mean([d["remote_total_tokens"] for d in details])
                
                time_breakdown = f"{avg_local_time:.2f}s/{avg_remote_time:.2f}s"
                token_breakdown = f"{avg_local_tokens:.0f}/{avg_remote_tokens:.0f}"
                
                print(f"{model_name:<40} {avg_exec_time:.2f}s{'':<10} {time_breakdown:<20} {token_breakdown:<15}")

def main():
    """Main function to run the performance tests."""
    
    # Check for OpenAI API key in environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        print("⚠️  OpenAI API key not found in environment variables.")
        print("You can either:")
        print("1. Set the OPENAI_API_KEY environment variable")
        print("2. Provide it manually below (optional)")
        
        manual_key = input("\nEnter OpenAI API key (or press Enter to skip): ").strip()
        if manual_key:
            openai_key = manual_key
        else:
            print("⚠️  No OpenAI API key provided. Remote model comparisons will not work.")
    else:
        print("✅ Found OpenAI API key in environment variables.")
    
    # Available Lemonade models from your configuration
    lemonade_models = [
        "Llama-3.2-3B-Instruct-Hybrid",
        "Qwen2.5-0.5B-Instruct-CPU", 
        "Llama-3.2-1B-Instruct-Hybrid",
        "Phi-3-Mini-Instruct-Hybrid",
        "Qwen-1.5-7B-Chat-Hybrid",
        #"DeepSeek-R1-Distill-Llama-8B-Hybrid"
        "DeepSeek-R1-Distill-Qwen-7B-Hybrid"
    ]
        
    # Initialize tester
    tester = LemonadePerformanceTester(models=lemonade_models, openai_api_key=openai_key)
    
    # Run comprehensive tests with detailed per-iteration output
    results = tester.run_comprehensive_test(iterations=10)  # Adjust iterations as needed
    
    # Print final summary
    tester.print_final_summary(results)
    
    # Save detailed results
    #tester.save_results(results)
    
    print(f"\n✅ Testing complete! Check the saved JSON file (if saved) for full details.")

if __name__ == "__main__":
    main()
