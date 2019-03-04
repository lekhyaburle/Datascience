package com.fragma.model;

public class TeamScores {
	private String teamName;
	private int fours;
	private int noofSixes;
	private int total;
	public String getTeamName() {
		return teamName;
	}
	public void setTeamName(String teamName) {
		this.teamName = teamName;
	}
	public int getFours() {
		return fours;
	}
	public void setFours(int fours) {
		this.fours = fours;
	}
	public int getNoofSixes() {
		return noofSixes;
	}
	public void setNoofSixes(int noofSixes) {
		this.noofSixes = noofSixes;
	}
	public int getTotal() {
		return total;
	}
	public void setTotal(int total) {
		this.total = total;
	}
	public TeamScores() {
		super();
	}
	public TeamScores(String teamName, int fours, int noofSixes, int total) {
		super();
		this.teamName = teamName;
		this.fours = fours;
		this.noofSixes = noofSixes;
		this.total = total;
	}
	
}
