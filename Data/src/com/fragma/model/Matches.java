package com.fragma.model;

import java.util.Date;

public class Matches {

	private int matchId;
	private int season;
	private String city;
	private Date matchDate;
	private String Team1;
	private String Team2;
	private String tossWinner;
	private String tossDecision;
	private String result;
	private String Winner;
	public int getMatchId() {
		return matchId;
	}
	public void setMatchId(int matchId) {
		this.matchId = matchId;
	}
	public int getSeason() {
		return season;
	}
	public void setSeason(int season) {
		this.season = season;
	}
	public String getCity() {
		return city;
	}
	public void setCity(String city) {
		this.city = city;
	}
	public Date getMatchDate() {
		return matchDate;
	}
	public void setMatchDate(Date matchDate) {
		this.matchDate = matchDate;
	}
	public String getTeam1() {
		return Team1;
	}
	public void setTeam1(String team1) {
		Team1 = team1;
	}
	public String getTeam2() {
		return Team2;
	}
	public void setTeam2(String team2) {
		Team2 = team2;
	}
	public String getTossWinner() {
		return tossWinner;
	}
	public void setTossWinner(String tossWinner) {
		this.tossWinner = tossWinner;
	}
	public String getTossDecision() {
		return tossDecision;
	}
	public void setTossDecision(String tossDecision) {
		this.tossDecision = tossDecision;
	}
	public String getResult() {
		return result;
	}
	public void setResult(String result) {
		this.result = result;
	}
	public String getWinner() {
		return Winner;
	}
	public void setWinner(String winner) {
		Winner = winner;
	}
	public Matches() {
		super();
	}
	public Matches(int matchId, int season, String city, Date matchDate, String team1, String team2,
			String tossWinner, String tossDecision, String result, String winner) {
		super();
		this.matchId = matchId;
		this.season = season;
		this.city = city;
		this.matchDate = matchDate;
		this.Team1 = team1;
		this.Team2 = team2;
		this.tossWinner = tossWinner;
		this.tossDecision = tossDecision;
		this.result = result;
		this.Winner = winner;
	}
	
	
}
