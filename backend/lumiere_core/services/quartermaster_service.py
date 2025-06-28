# backend/lumiere_core/services/quartermaster_service.py

import json
import logging
import os
import tempfile
import subprocess
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import requests
import docker
from docker.errors import DockerException

from .bom_parser import get_bom_data, has_bom_data, TechStackBOM
from .crucible import validate_fix
from .cortex_service import load_cortex_data
from .llm_service import generate_text, TaskType

logger = logging.getLogger(__name__)

@dataclass
class Vulnerability:
    """Represents a security vulnerability in a dependency."""
    id: str
    severity: str  # low, medium, high, critical
    summary: str
    affected_package: str
    affected_versions: List[str]
    fixed_versions: List[str]
    published: str
    modified: str
    database_specific: Dict[str, Any]
    references: List[str]
    aliases: List[str]
    
@dataclass
class LicenseViolation:
    """Represents a license compliance violation."""
    package_name: str
    package_version: str
    license: str
    violation_type: str  # denied, restricted, unknown
    policy_rule: str
    recommendation: str

@dataclass
class UpgradeSimulationReport:
    """Represents the result of an upgrade simulation."""
    package_name: str
    current_version: str
    target_version: str
    simulation_successful: bool
    test_results: Dict[str, Any]
    breaking_changes: List[str]
    compatibility_score: float
    recommendation: str
    error_details: Optional[str] = None

@dataclass
class CompliancePolicy:
    """Represents a license compliance policy."""
    allowed_licenses: List[str]
    denied_licenses: List[str]
    restricted_licenses: List[str]  # require approval
    policy_name: str
    created_at: str

class QuartermasterService:
    """
    The Quartermaster's Inventory - Advanced dependency management service.
    Evolved from the Merchant's passive BOM into proactive supply-chain guardian.
    """
    
    def __init__(self, cloned_repos_dir: Path):
        self.cloned_repos_dir = Path(cloned_repos_dir)
        self.osv_api_base = "https://api.osv.dev/v1"
        self.docker_client = None
        
        # Initialize Docker client if available
        try:
            self.docker_client = docker.from_env()
            logger.info("Quartermaster: Docker client initialized successfully")
        except DockerException:
            logger.warning("Quartermaster: Docker not available - upgrade simulations will be limited")
    
    def get_dashboard_health(self, repo_id: str) -> Dict[str, Any]:
        """
        Generate a high-level supply-chain health dashboard.
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            Dictionary with health indicators and summary
        """
        try:
            logger.info(f"Quartermaster: Generating dashboard for {repo_id}")
            
            # Load BOM data
            bom_data = get_bom_data(repo_id)
            if not bom_data:
                return {"error": "BOM data not found for repository"}
            
            # Get vulnerabilities
            vulnerabilities = self.check_vulnerabilities(bom_data)
            
            # Check license compliance with default policy
            default_policy = CompliancePolicy(
                allowed_licenses=["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "ISC"],
                denied_licenses=["GPL-3.0", "AGPL-3.0", "SSPL-1.0"],
                restricted_licenses=["GPL-2.0", "LGPL-3.0"],
                policy_name="default_policy",
                created_at=datetime.now().isoformat()
            )
            
            license_violations = self.check_license_compliance(bom_data, default_policy)
            
            # Calculate health scores
            security_health = self._calculate_security_health(vulnerabilities)
            license_health = self._calculate_license_health(license_violations)
            freshness_health = self._calculate_freshness_health(bom_data)
            
            # Overall health
            overall_health = min(security_health, license_health, freshness_health)
            health_color = self._get_health_color(overall_health)
            
            dashboard = {
                "repository": repo_id,
                "overall_health": {
                    "score": overall_health,
                    "color": health_color,
                    "status": self._get_health_status(overall_health)
                },
                "security_health": {
                    "score": security_health,
                    "color": self._get_health_color(security_health),
                    "critical_vulns": len([v for v in vulnerabilities if v.severity == "critical"]),
                    "high_vulns": len([v for v in vulnerabilities if v.severity == "high"]),
                    "total_vulns": len(vulnerabilities)
                },
                "license_health": {
                    "score": license_health,
                    "color": self._get_health_color(license_health),
                    "violations": len(license_violations),
                    "denied_licenses": len([v for v in license_violations if v.violation_type == "denied"])
                },
                "dependency_freshness": {
                    "score": freshness_health,
                    "color": self._get_health_color(freshness_health),
                    "total_dependencies": bom_data.get("summary", {}).get("total_dependencies", 0),
                    "ecosystems": bom_data.get("summary", {}).get("ecosystems", [])
                },
                "last_scan": datetime.now().isoformat(),
                "recommendations": self._generate_dashboard_recommendations(
                    vulnerabilities, license_violations, bom_data
                )
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Quartermaster: Error generating dashboard for {repo_id}: {e}")
            return {"error": f"Failed to generate dashboard: {str(e)}"}
    
    def check_vulnerabilities(self, bom_data: Dict[str, Any]) -> List[Vulnerability]:
        """
        Check dependencies against OSV vulnerability database.
        
        Args:
            bom_data: Bill of Materials data
            
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        try:
            # Process all dependency categories
            for category, deps in bom_data.get("dependencies", {}).items():
                for dep in deps:
                    package_name = dep.get("name")
                    version = dep.get("version", "").replace("^", "").replace("~", "").replace(">=", "").replace("<=", "").replace("==", "").strip()
                    ecosystem = dep.get("ecosystem", "").lower()
                    
                    if not package_name or not version or version == "any":
                        continue
                    
                    # Map ecosystem to OSV format
                    osv_ecosystem = self._map_ecosystem_to_osv(ecosystem)
                    if not osv_ecosystem:
                        continue
                    
                    # Query OSV API
                    vulns = self._query_osv_api(package_name, version, osv_ecosystem)
                    vulnerabilities.extend(vulns)
            
            logger.info(f"Quartermaster: Found {len(vulnerabilities)} vulnerabilities")
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Quartermaster: Error checking vulnerabilities: {e}")
            return []
    
    def simulate_upgrade(self, repo_id: str, dependency_name: str, target_version: str) -> UpgradeSimulationReport:
        """
        Simulate upgrading a dependency and test compatibility.
        
        Args:
            repo_id: Repository identifier
            dependency_name: Name of dependency to upgrade
            target_version: Target version to upgrade to
            
        Returns:
            Simulation report with results and recommendations
        """
        try:
            logger.info(f"Quartermaster: Simulating upgrade of {dependency_name} to {target_version} for {repo_id}")
            
            repo_path = self.cloned_repos_dir / repo_id
            if not repo_path.exists():
                return UpgradeSimulationReport(
                    package_name=dependency_name,
                    current_version="unknown",
                    target_version=target_version,
                    simulation_successful=False,
                    test_results={},
                    breaking_changes=[],
                    compatibility_score=0.0,
                    recommendation="Repository not found",
                    error_details="Repository directory does not exist"
                )
            
            # Create temporary copy of repository
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_repo = Path(temp_dir) / "temp_repo"
                shutil.copytree(repo_path, temp_repo)
                
                # Find and modify manifest files
                current_version = self._get_current_dependency_version(temp_repo, dependency_name)
                manifest_modified = self._modify_dependency_version(temp_repo, dependency_name, target_version)
                
                if not manifest_modified:
                    return UpgradeSimulationReport(
                        package_name=dependency_name,
                        current_version=current_version,
                        target_version=target_version,
                        simulation_successful=False,
                        test_results={},
                        breaking_changes=[],
                        compatibility_score=0.0,
                        recommendation="Could not modify dependency in manifest files",
                        error_details="Dependency not found in manifest files"
                    )
                
                # Run tests in modified environment
                test_results = self._run_tests_in_environment(temp_repo)
                
                # Analyze compatibility
                breaking_changes = self._analyze_breaking_changes(dependency_name, current_version, target_version)
                compatibility_score = self._calculate_compatibility_score(test_results, breaking_changes)
                
                # Generate recommendation
                recommendation = self._generate_upgrade_recommendation(
                    dependency_name, current_version, target_version, 
                    compatibility_score, test_results, breaking_changes
                )
                
                return UpgradeSimulationReport(
                    package_name=dependency_name,
                    current_version=current_version,
                    target_version=target_version,
                    simulation_successful=test_results.get("success", False),
                    test_results=test_results,
                    breaking_changes=breaking_changes,
                    compatibility_score=compatibility_score,
                    recommendation=recommendation
                )
                
        except Exception as e:
            logger.error(f"Quartermaster: Error simulating upgrade: {e}")
            return UpgradeSimulationReport(
                package_name=dependency_name,
                current_version="unknown",
                target_version=target_version,
                simulation_successful=False,
                test_results={},
                breaking_changes=[],
                compatibility_score=0.0,
                recommendation="Simulation failed due to internal error",
                error_details=str(e)
            )
    
    def check_license_compliance(self, bom_data: Dict[str, Any], policy: CompliancePolicy) -> List[LicenseViolation]:
        """
        Check dependencies against license compliance policy.
        
        Args:
            bom_data: Bill of Materials data
            policy: License compliance policy
            
        Returns:
            List of license violations
        """
        violations = []
        
        try:
            for category, deps in bom_data.get("dependencies", {}).items():
                for dep in deps:
                    package_name = dep.get("name")
                    package_version = dep.get("version", "unknown")
                    license_name = dep.get("license")
                    
                    if not license_name:
                        violations.append(LicenseViolation(
                            package_name=package_name,
                            package_version=package_version,
                            license="unknown",
                            violation_type="unknown",
                            policy_rule="License information required",
                            recommendation="Investigate and document license for this dependency"
                        ))
                        continue
                    
                    # Check against policy
                    if license_name in policy.denied_licenses:
                        violations.append(LicenseViolation(
                            package_name=package_name,
                            package_version=package_version,
                            license=license_name,
                            violation_type="denied",
                            policy_rule=f"License '{license_name}' is explicitly denied",
                            recommendation=f"Remove or replace {package_name} - license incompatible with policy"
                        ))
                    elif license_name in policy.restricted_licenses:
                        violations.append(LicenseViolation(
                            package_name=package_name,
                            package_version=package_version,
                            license=license_name,
                            violation_type="restricted",
                            policy_rule=f"License '{license_name}' requires approval",
                            recommendation=f"Seek legal approval before using {package_name}"
                        ))
                    elif license_name not in policy.allowed_licenses:
                        violations.append(LicenseViolation(
                            package_name=package_name,
                            package_version=package_version,
                            license=license_name,
                            violation_type="unknown",
                            policy_rule=f"License '{license_name}' not in approved list",
                            recommendation=f"Review license '{license_name}' and update policy as needed"
                        ))
            
            logger.info(f"Quartermaster: Found {len(violations)} license violations")
            return violations
            
        except Exception as e:
            logger.error(f"Quartermaster: Error checking license compliance: {e}")
            return []
    
    def generate_risk_report(self, repo_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report for management.
        
        Args:
            repo_id: Repository identifier
            
        Returns:
            Risk report in human-readable format
        """
        try:
            dashboard = self.get_dashboard_health(repo_id)
            if "error" in dashboard:
                return dashboard
            
            # Load additional context
            bom_data = get_bom_data(repo_id)
            metadata = load_cortex_data(repo_id)
            
            # Generate executive summary using LLM
            summary_prompt = f"""
            You are a cybersecurity analyst preparing an executive summary for a software project's supply chain risks.
            
            Project: {repo_id}
            Security Health: {dashboard['security_health']['score']}/100
            License Compliance: {dashboard['license_health']['score']}/100
            Dependencies: {dashboard['dependency_freshness']['total_dependencies']}
            Critical Vulnerabilities: {dashboard['security_health']['critical_vulns']}
            High Vulnerabilities: {dashboard['security_health']['high_vulns']}
            License Violations: {dashboard['license_health']['violations']}
            
            Write a concise 2-3 paragraph executive summary focusing on:
            1. Overall risk level and key concerns
            2. Immediate actions required
            3. Strategic recommendations for risk mitigation
            
            Use business language appropriate for technical management.
            """
            
            executive_summary = generate_text(summary_prompt, task_type=TaskType.ANALYSIS)
            
            risk_report = {
                "repository": repo_id,
                "report_date": datetime.now().isoformat(),
                "executive_summary": executive_summary,
                "risk_metrics": {
                    "overall_risk_score": 100 - dashboard["overall_health"]["score"],
                    "security_risk": 100 - dashboard["security_health"]["score"],
                    "compliance_risk": 100 - dashboard["license_health"]["score"],
                    "operational_risk": 100 - dashboard["dependency_freshness"]["score"]
                },
                "key_findings": {
                    "critical_vulnerabilities": dashboard["security_health"]["critical_vulns"],
                    "high_vulnerabilities": dashboard["security_health"]["high_vulns"],
                    "license_violations": dashboard["license_health"]["violations"],
                    "total_dependencies": dashboard["dependency_freshness"]["total_dependencies"],
                    "primary_language": bom_data.get("summary", {}).get("primary_language", "unknown")
                },
                "recommendations": dashboard["recommendations"],
                "next_review_date": (datetime.now() + timedelta(days=30)).isoformat()
            }
            
            return risk_report
            
        except Exception as e:
            logger.error(f"Quartermaster: Error generating risk report: {e}")
            return {"error": f"Failed to generate risk report: {str(e)}"}
    
    # Private helper methods
    
    def _calculate_security_health(self, vulnerabilities: List[Vulnerability]) -> float:
        """Calculate security health score based on vulnerabilities."""
        if not vulnerabilities:
            return 100.0
        
        # Weight vulnerabilities by severity
        severity_weights = {"critical": 40, "high": 20, "medium": 5, "low": 1}
        total_weight = sum(severity_weights.get(v.severity, 1) for v in vulnerabilities)
        
        # Scale inversely with total weight (more severe issues = lower score)
        return max(0.0, 100.0 - min(100.0, total_weight))
    
    def _calculate_license_health(self, violations: List[LicenseViolation]) -> float:
        """Calculate license compliance health score."""
        if not violations:
            return 100.0
        
        # Weight violations by type
        violation_weights = {"denied": 50, "restricted": 20, "unknown": 10}
        total_weight = sum(violation_weights.get(v.violation_type, 10) for v in violations)
        
        return max(0.0, 100.0 - min(100.0, total_weight))
    
    def _calculate_freshness_health(self, bom_data: Dict[str, Any]) -> float:
        """Calculate dependency freshness health score."""
        # This is a simplified implementation
        # In practice, you'd check actual dependency ages and update frequencies
        return 85.0  # Placeholder score
    
    def _get_health_color(self, score: float) -> str:
        """Convert health score to color indicator."""
        if score >= 80:
            return "green"
        elif score >= 60:
            return "yellow"
        else:
            return "red"
    
    def _get_health_status(self, score: float) -> str:
        """Convert health score to status text."""
        if score >= 80:
            return "Healthy"
        elif score >= 60:
            return "Needs Attention"
        else:
            return "Critical Issues"
    
    def _generate_dashboard_recommendations(self, vulnerabilities: List[Vulnerability], 
                                         violations: List[LicenseViolation], 
                                         bom_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        if vulnerabilities:
            critical_vulns = [v for v in vulnerabilities if v.severity == "critical"]
            if critical_vulns:
                recommendations.append(f"URGENT: Address {len(critical_vulns)} critical security vulnerabilities immediately")
            
            high_vulns = [v for v in vulnerabilities if v.severity == "high"]
            if high_vulns:
                recommendations.append(f"Update {len(high_vulns)} dependencies with high-severity vulnerabilities")
        
        if violations:
            denied_violations = [v for v in violations if v.violation_type == "denied"]
            if denied_violations:
                recommendations.append(f"Remove or replace {len(denied_violations)} dependencies with denied licenses")
        
        # Add general recommendations
        recommendations.extend([
            "Enable automated dependency scanning in CI/CD pipeline",
            "Schedule monthly dependency health reviews",
            "Implement dependency update automation where possible"
        ])
        
        return recommendations
    
    def _map_ecosystem_to_osv(self, ecosystem: str) -> Optional[str]:
        """Map internal ecosystem names to OSV database format."""
        mapping = {
            "python": "PyPI",
            "javascript": "npm",
            "java": "Maven",
            "go": "Go",
            "rust": "crates.io",
            "php": "Packagist",
            "ruby": "RubyGems",
            "nuget": "NuGet"
        }
        return mapping.get(ecosystem.lower())
    
    def _query_osv_api(self, package_name: str, version: str, ecosystem: str) -> List[Vulnerability]:
        """Query OSV API for vulnerabilities in a specific package version."""
        try:
            # Query OSV API
            query_url = f"{self.osv_api_base}/query"
            payload = {
                "package": {
                    "name": package_name,
                    "ecosystem": ecosystem
                },
                "version": version
            }
            
            response = requests.post(query_url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            vulnerabilities = []
            
            for vuln_data in result.get("vulns", []):
                vulnerability = Vulnerability(
                    id=vuln_data.get("id", "unknown"),
                    severity=self._extract_severity(vuln_data),
                    summary=vuln_data.get("summary", "No summary available"),
                    affected_package=package_name,
                    affected_versions=self._extract_affected_versions(vuln_data),
                    fixed_versions=self._extract_fixed_versions(vuln_data),
                    published=vuln_data.get("published", ""),
                    modified=vuln_data.get("modified", ""),
                    database_specific=vuln_data.get("database_specific", {}),
                    references=[ref.get("url", "") for ref in vuln_data.get("references", [])],
                    aliases=vuln_data.get("aliases", [])
                )
                vulnerabilities.append(vulnerability)
            
            return vulnerabilities
            
        except Exception as e:
            logger.warning(f"Quartermaster: Error querying OSV API for {package_name}: {e}")
            return []
    
    def _extract_severity(self, vuln_data: Dict[str, Any]) -> str:
        """Extract severity from vulnerability data."""
        # Try CVSS score first
        if "severity" in vuln_data:
            for severity_info in vuln_data["severity"]:
                if severity_info.get("type") == "CVSS_V3":
                    score = severity_info.get("score", 0)
                    if score >= 9.0:
                        return "critical"
                    elif score >= 7.0:
                        return "high"
                    elif score >= 4.0:
                        return "medium"
                    else:
                        return "low"
        
        # Fallback to database-specific severity
        db_specific = vuln_data.get("database_specific", {})
        severity = db_specific.get("severity", "medium").lower()
        return severity if severity in ["critical", "high", "medium", "low"] else "medium"
    
    def _extract_affected_versions(self, vuln_data: Dict[str, Any]) -> List[str]:
        """Extract affected versions from vulnerability data."""
        affected = []
        for affected_item in vuln_data.get("affected", []):
            versions = affected_item.get("versions", [])
            affected.extend(versions)
        return affected
    
    def _extract_fixed_versions(self, vuln_data: Dict[str, Any]) -> List[str]:
        """Extract fixed versions from vulnerability data."""
        fixed = []
        for affected_item in vuln_data.get("affected", []):
            ranges = affected_item.get("ranges", [])
            for range_item in ranges:
                events = range_item.get("events", [])
                for event in events:
                    if "fixed" in event:
                        fixed.append(event["fixed"])
        return fixed
    
    def _get_current_dependency_version(self, repo_path: Path, dependency_name: str) -> str:
        """Find the current version of a dependency in the repository."""
        # This is a simplified implementation
        # In practice, you'd parse various manifest files
        return "current_version"  # Placeholder
    
    def _modify_dependency_version(self, repo_path: Path, dependency_name: str, target_version: str) -> bool:
        """Modify dependency version in manifest files."""
        # This is a simplified implementation
        # In practice, you'd modify package.json, requirements.txt, etc.
        return True  # Placeholder
    
    def _run_tests_in_environment(self, repo_path: Path) -> Dict[str, Any]:
        """Run tests in the modified repository environment."""
        # This is a simplified implementation
        # In practice, you'd use Docker or virtual environments
        return {
            "success": True,
            "tests_run": 10,
            "tests_passed": 8,
            "tests_failed": 2,
            "coverage": 85.0
        }  # Placeholder
    
    def _analyze_breaking_changes(self, dependency_name: str, current_version: str, target_version: str) -> List[str]:
        """Analyze potential breaking changes between versions."""
        # This is a simplified implementation
        # In practice, you'd check changelogs, semver, etc.
        return ["API endpoint deprecated", "Function signature changed"]  # Placeholder
    
    def _calculate_compatibility_score(self, test_results: Dict[str, Any], breaking_changes: List[str]) -> float:
        """Calculate compatibility score based on test results and breaking changes."""
        base_score = 100.0
        
        # Deduct for test failures
        if test_results.get("tests_failed", 0) > 0:
            failure_rate = test_results["tests_failed"] / test_results.get("tests_run", 1)
            base_score -= (failure_rate * 50)
        
        # Deduct for breaking changes
        base_score -= (len(breaking_changes) * 10)
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_upgrade_recommendation(self, dependency_name: str, current_version: str, 
                                       target_version: str, compatibility_score: float,
                                       test_results: Dict[str, Any], breaking_changes: List[str]) -> str:
        """Generate upgrade recommendation based on analysis."""
        if compatibility_score >= 90:
            return "RECOMMENDED: Upgrade appears safe with minimal risk"
        elif compatibility_score >= 70:
            return "CAUTION: Review breaking changes and test failures before upgrading"
        else:
            return "NOT RECOMMENDED: Significant compatibility issues detected"