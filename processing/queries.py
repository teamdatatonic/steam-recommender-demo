PERCENTILE_SQL = """
    SELECT
        PERCENTILE_CONT(playtime, 0.25) OVER() AS pctl_25,
        PERCENTILE_CONT(playtime, 0.5) OVER() AS pctl_50,
        PERCENTILE_CONT(playtime, 0.75) OVER() AS pctl_75,
        PERCENTILE_CONT(playtime, 0.95) OVER() AS pctl_95
    FROM `{model_repo}`
    LIMIT 1
"""

EVALUATION_SET_SQL = """
    WITH users AS(
        SELECT DISTINCT(steamid) FROM `{model_repo}`
    ),

    games AS(
        SELECT DISTINCT(appid) FROM `{model_repo}`
    ),

    user_subset AS(
        SELECT
            steamid
        FROM users
        LIMIT 5
    )

    SELECT a.appid, a.steamid, IFNULL(playtime, 0) AS playtime
    FROM (
        SELECT appid, steamid FROM user_subset CROSS JOIN games
    ) a
    LEFT JOIN `{model_repo}` b
    ON a.appid = b.appid
    AND a.steamid = b.steamid
"""

FAKE_INTERACTIONS_SQL = """
    WITH all_pairs AS(
        SELECT appid, steamid, RAND() AS x
        FROM `Steam.users` CROSS JOIN `Steam.games`
    )

    SELECT
      all_pairs.steamid, all_pairs.appid, 0 as `{label}`
    FROM
      all_pairs
      LEFT JOIN `{model_repo}`
      ON all_pairs.steamid = `{model_repo}`.steamid
      AND all_pairs.appid = `{model_repo}`.appid
    WHERE playtime IS NULL
    AND x < 0.01
"""
