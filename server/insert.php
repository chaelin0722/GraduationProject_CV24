<!-- $connection = mysqli_connect("localhost", "cv24", "1234", );

  -->
<?php

    error_reporting(E_ALL);
    ini_set('display_errors',1);

    include('dbcon.php');

    if( ($_SERVER['REQUEST_METHOD'] == 'POST') && isset($_POST['submit']))
        {
            $id = $_POST['id'];
            $situation = $_POST['situation'];
            $time = $_POST['time'];
            $latitude = $_POST['latitude'];
            $longitude = $_POST['longitude'];
            if(empty($situation)){
                $errMSG = "상황을 입력하세요.";
            }
            else if(empty($time)){
                $errMSG = "YYYY-MM-DD HH:MM:SS 순서대로 입력하세요.";
            }

            if(!isset($errMSG))
            {
                try{
                    $stmt = $con->prepare('INSERT INTO test(id, situation,time, latitude, longitude)
                                    VALUES(:id, :situation, :time, :latitude, :longitude)');
                    $stmt->bindParam(':id', $id);
                    $stmt->bindParam(':situation', $situation);
                    $stmt->bindParam(':time', $time);
                    $stmt->bindParam(':latitude', $latitude);
                    $stmt->bindParam(':longitude', $longitude);

                    if($stmt->execute())
                    {
                        $successMSG = "새로운 사용자를 추가했습니다.";
                    }
                    else
                    {
                        $errMSG = "사용자 추가 에러";
                    }

                } catch(PDOException $e) {
                    die("Database error: " . $e->getMessage());
                }
            }
        }
?>

<html>
    <body>
        <?php
        if (isset($errMSG)) echo $errMSG;
        if (isset($successMSG)) echo $successMSG;
        ?>

        <form action="<?php $_PHP_SELF ?>" method="POST">
           id: <input type = "text" name = "id" />
           Situation: <input type = "text" name = "situation" />
           datetime: <input type = "text" name = "time" />
           latitude: <input type = "text" name = "latitude" />
           longitude: <input type = "text" name = "longitude" />
           <input type = "submit" name = "submit"/>
        </form>

    </body>
</html>